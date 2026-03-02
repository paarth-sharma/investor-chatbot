# Hero MotoCorp Financial Chatbot — Architecture & Data Flow

## What This Is

A **Hero MotoCorp financial analysis chatbot**. You can ask it questions like "What was net margin in Q3 FY25?" or "Compare revenue Q1 FY24 vs Q1 FY23" and get accurate, deterministic answers grounded in 15 quarters of actual earnings report data (Q1 FY23 → Q4 FY26).

---

## The Services (Docker Compose)

Five containers, all on an internal `hero-net` network:

| Container | Port | Role |
|---|---|---|
| `hero-graph-db` | 7474, 7687 | Neo4j — stores both graph data and vector embeddings |
| `hero-quant-engine` | 8000 | FastAPI — pure deterministic math, zero ML |
| `hero-llm-extractor` | 11434 | Ollama — serves two LLM models (Qwen2.5:3b variants) |
| `hero-orchestrator` | — | Runs `ingest.py` and `query.py` on demand |
| `hero-chatbot` | 8501 | FastAPI — the chat API (`chat.py`) |

---

## Phase 1: Ingestion (Offline, One-Time)

You run `docker exec hero-orchestrator python ingest.py`. It processes all 15 JSON files sequentially.

### What the raw data looks like

Each JSON file is an array of elements from **Unstructured.io**'s VLM pipeline (it ran a vision-language model on the PDF). Most elements have a `text` field (raw OCR). Exactly **one** element has `metadata.extracted_data` — a structured dict of financial numbers already extracted by Unstructured.io.

The problem: `extracted_data` schemas are **inconsistent** across quarters — some are flat, some nested, key names vary (e.g. `profit_after_tax` vs `profit_after_tax_crore`).

### For each file, two pipelines run:

**Pipeline A — Structured Data (CPU only, no LLM):**

1. Parse filename → `Q1-apr-jun-2022.json` → `Q1 FY23` (Indian FY offset logic at `ingest.py:49`)
2. Find the `extracted_data` element
3. Detect financial sections (`find_financial_sections()` at `ingest.py:193`) — finds standalone vs consolidated results regardless of nesting
4. Recursively flatten the dict with `flatten_financial_data()` at `ingest.py:247` — handles nested period scopes like `quarter_results.revenue` vs `nine_months_ended.revenue`
5. Strip unit suffixes from keys: `_crore` → unit="Cr", `_lakhs` → unit="Lakh" (`ingest.py:150`)
6. Fuzzy-match ~50 synonym mappings to standardize keys: `profit_after_tax` → `"Net Income"`, `finance_costs` → `"Interest Expense"`, etc. (`ingest.py:75`)
7. Extract extra metrics from other sections (highlights, dividends, equity) via `extract_extra_metrics()` at `ingest.py:318`
8. Write Neo4j nodes: `File -[:REPORTED]-> DataPoint -[:IS_METRIC]-> MetricDefinition`

**Pipeline B — Text Chunks (CPU parse + GPU embed):**

1. Collect all `text` fields (raw OCR from the PDF)
2. Strip HTML tags (BeautifulSoup), normalize whitespace (`ingest.py:381`)
3. Chunk into ~2000-char windows with 300-char overlap, breaking at sentence boundaries (`ingest.py:390`)
4. Batch-embed with **BGE-Small** (BAAI/bge-small-en-v1.5, 384-dim) on the RTX 4060 — batch size 64, ~1GB VRAM (`ingest.py:438`)
5. Write Neo4j nodes: `File -[:HAS_CHUNK]-> Chunk` with the full embedding vector stored on each node

**Neo4j setup:**
- Vector index `financial_vectors` on `Chunk.embedding` (cosine similarity, 384-dim)
- Uniqueness constraint on `MetricDefinition.name`

---

## Phase 2: Query Time (Live, Per Request)

A user sends `POST /chat` with `{"message": "What was Hero MotoCorp revenue in Q1 FY24?"}` to port 8501.

### Step 1 — Pre-processing (Python, no LLM)
`preprocess_calendar_to_fy()` at `chat.py:209` converts calendar year references before the LLM sees them. "Q3 of 2022" → "Q3 FY23". This means the small 3b model never has to do date arithmetic.

### Step 2 — Intent Classification (`finma` model, GPU)
`llm_parse_query()` at `chat.py:228` calls the **`finma` model** (Qwen2.5:3b with a custom Modelfile: `temp=0, seed=42, top_k=1`, forced JSON output). It gets passed:
- The user's question
- The **exact list of available periods** fetched live from Neo4j
- The **exact list of available metric names** fetched live from Neo4j

Because the LLM is given the exact strings it must choose from, it **cannot hallucinate metric or period names**. It returns JSON like:
```json
{"query_type": "lookup", "metrics": ["Revenue"], "periods": ["Q1 FY24"], "quant_endpoint": null}
```

Query types: `lookup | compare | trend | calculate | general`

### Step 3 — Data Retrieval (CPU, Neo4j graph)
`fetch_metric()` at `chat.py:124` queries Neo4j. It:
- Tries metric aliases first (e.g. "Revenue" → also tries "Revenue from Operations", "Total Income")
- Prefers **Consolidated** over Standalone, high-confidence over low-confidence
- Falls back to a fuzzy `toLower() CONTAINS` search if exact name fails

### Step 4 — RAG Search (GPU, always runs for lookup/compare)
`rag_search()` at `chat.py:178`:
- Embeds the query with BGE-Small on GPU
- Calls `db.index.vector.queryNodes('financial_vectors', 5, $embedding)` in Neo4j
- Returns top-5 text chunks by cosine similarity
- These chunks are passed to the answer formatter as qualitative context (e.g., "product launches, market share notes")

### Step 5 — Quant Engine Call (CPU, only for calculate/compare/trend)
`call_quant_engine()` at `chat.py:196` sends a `POST` to `http://quant-engine:8000/<endpoint>`. Pure Python math — no ML. 30 endpoints covering profitability, liquidity, leverage, efficiency, and valuation ratios.

Example: `POST /net-margin {"net_income": 1202.84, "revenue": 10210.78}` → `{"result": 11.78, "unit": "%"}`

### Step 6 — Answer Formatting (`qwen2.5:3b` base model, GPU)
`llm_format_answer()` at `chat.py:313` uses the **base qwen2.5:3b** (NOT finma — finma outputs JSON, you need natural language here). The prompt gives it:
- The user's question
- The verified numbers from Neo4j
- The calculation results from quant engine
- The top-3 RAG text chunks for qualitative context

Strict rules in the prompt: no invented numbers, Indian format (Cr/Lakh), 2-4 sentences max.

### Query type routing summary:

| Type | What happens |
|---|---|
| `lookup` | Graph fetch → RAG → LLM format |
| `compare` | Graph fetch (2 periods) → quant `/growth` → RAG → LLM format |
| `trend` | Graph fetch (all periods) → quant `/growth` per consecutive pair → LLM format |
| `calculate` | Graph fetch (all required metrics) → quant endpoint → RAG if missing → LLM format |
| `general` | RAG only → LLM format |

---

## Chatbot API Endpoints (`chat.py`)

All served on port 8501.

● Since port 8501 is mapped to the host in docker-compose.yml, you don't need docker exec — you can hit it directly with curl from your machine. Piping through python3 -m json.tool pretty-prints the JSON:                         
1. Natural language query
```
  curl -s -X POST http://localhost:8501/chat -H "Content-Type: application/json" -d '{"message": "What was revenue in Q1 FY24?"}' | python3 -m json.tool
```
2. List all metrics (with period coverage + count)
```
  curl -s http://localhost:8501/metrics | python3 -m json.tool
```
3. List all available periods
```
  curl -s http://localhost:8501/periods | python3 -m json.tool
```
4. Debug raw DataPoints for a specific period
```
  curl -s "http://localhost:8501/debug/data/Q1%20FY24" | python3 -m json.tool
```

  Note the URL-encoded space (%20) in the period string — Q1 FY24 → Q1%20FY24. Swap in whichever period you want to inspect.

  If you do need to run these from inside a container (e.g. the network isn't accessible from your host), exec into the orchestrator which is on the same Docker
   network:
```
  docker exec hero-orchestrator curl -s http://hero-chatbot:8501/periods
```

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | Main chat interface — accepts `{message, history}`, returns `{answer, data, calculations, sources, debug}` |
| `GET` | `/` | Health check — returns service name and version |
| `GET` | `/metrics` | Lists every `MetricDefinition` in the graph with the periods it covers and a data point count. Use this after ingestion to verify all metrics were captured and to see their period coverage. (`chat.py:574`) |
| `GET` | `/periods` | Returns all distinct period strings available in the graph (e.g. `["Q1 FY23", "Q2 FY23", ...]`). Useful for knowing what date range the database covers. (`chat.py:587`) |
| `GET` | `/debug/data/{period}` | Returns every raw `DataPoint` for a given period — metric name, value, unit, context (Standalone/Consolidated), original key from the JSON, and source path within `extracted_data`. Use this to inspect exactly what was ingested for a specific quarter and catch normalisation errors. (`chat.py:592`) |

---

## `query.py` — Batch Report Mode

A CLI alternative to chat. `python query.py "Q3 FY25"`:
- Fetches all 20 standard metrics from the graph
- Falls back to RAG + LLM extraction for any missing metric (at `query.py:94` — this is the only place the LLM is used for extraction, and only as a last resort)
- Calls every applicable quant engine endpoint
- Prints a formatted table — no natural language response

---

## The Two LLM Models

| Model | Used for | Why separate |
|---|---|---|
| `finma` | Intent classification, JSON extraction | Custom Modelfile forces JSON output, `temp=0`, `seed=42` — deterministic, structured |
| `qwen2.5:3b` | Answer formatting | Base model for natural language, same determinism settings |

The `finma` model is created at container startup: `ollama create finma -f /Modelfile` on the Qwen2.5:3b base.

---

## Key Design Principle

**The LLM is glue, not a processor.** It only does two things: classify intent and format English sentences. Every number in every answer comes from either Neo4j (exact graph lookup) or the quant engine (pure math). The LLM is never trusted with arithmetic or data retrieval.
