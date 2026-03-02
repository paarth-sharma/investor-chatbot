# RAG Financial Chatbot — Implementation Plan

## Problem

The current `ingest.py` sends every text chunk (~100MB across 15 JSON files) through Qwen2.5 for LLM extraction, maxing out the 4060 Max-Q GPU. This is unnecessary because each JSON file already contains a clean `extracted_data` dict with structured financial metrics from Unstructured.io's VLM pipeline.

**Secondary problem**: The `extracted_data` schemas are inconsistent across files — flat vs nested, different key naming conventions (`profit_after_tax` vs `profit_after_tax_crore`), different nesting depths per quarter.

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION (Offline)                      │
│                                                              │
│  JSON Files ──┬── extracted_data ──→ [CPU] Normalizer        │
│               │   (structured)       Fuzzy key matching      │
│               │                      Schema flattening       │
│               │                      Period extraction        │
│               │                          │                    │
│               │                          ▼                    │
│               │                    Neo4j Graph DB             │
│               │                    (DataPoints + Metrics)     │
│               │                                              │
│               └── text elements ──→ [CPU] Chunker            │
│                   (raw OCR)         HTML strip, 300 tokens    │
│                                     15% overlap, metadata    │
│                                          │                    │
│                                          ▼                    │
│                                    [GPU] BGE-Small            │
│                                    Batch embed (~1GB VRAM)    │
│                                          │                    │
│                                          ▼                    │
│                                    Neo4j Vector Index         │
│                                    (Chunk embeddings)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY TIME (Online)                        │
│                                                              │
│  User: "What was net margin in Q3 FY25?"                     │
│           │                                                  │
│           ▼                                                  │
│  [GPU] Qwen2.5:3b — Intent Classification (~2-3GB VRAM)     │
│     → intent: "calculate"                                    │
│     → metrics_needed: ["NetIncome", "Revenue"]               │
│     → period: "Q3 FY25"                                      │
│     → quant_endpoint: "net-margin"                           │
│           │                                                  │
│           ▼                                                  │
│  [CPU] Neo4j Graph Lookup (structured data, fast)            │
│     → Revenue: 10210.78 Cr                                   │
│     → NetIncome: 1202.84 Cr                                  │
│           │                                                  │
│     (if metric missing) ──→ [GPU] Vector Search              │
│                               BGE-Small embed query           │
│                               Cosine similarity on chunks    │
│                               Extract value from top chunk   │
│           │                                                  │
│           ▼                                                  │
│  [CPU] Quant Engine — Deterministic Math                     │
│     POST /net-margin {net_income: 1202.84, revenue: 10210.78}│
│     → {result: 11.78, unit: "%"}                             │
│           │                                                  │
│           ▼                                                  │
│  [GPU] Qwen2.5:3b — Format Answer                           │
│     "The net profit margin for Q3 FY25 was 11.78%,           │
│      based on PAT of ₹1,202.84 Cr on revenue of             │
│      ₹10,210.78 Cr."                                         │
│           │                                                  │
│           ▼                                                  │
│  Response to User                                            │
└─────────────────────────────────────────────────────────────┘
```

## Hardware Load Distribution

| Component | Device | VRAM/RAM | When |
|-----------|--------|----------|------|
| JSON normalizer | CPU (i7 13th gen) | ~200MB RAM | Ingestion |
| Text chunking + HTML cleaning | CPU | ~500MB RAM | Ingestion |
| BGE-Small embeddings (batch) | GPU (4060 Max-Q) | ~1GB VRAM | Ingestion |
| Neo4j graph operations | CPU | ~1GB RAM | Both |
| Qwen2.5:3b intent parsing | GPU | ~2-3GB VRAM | Query time |
| BGE-Small query embedding | GPU | ~1GB VRAM | Query time |
| Quant engine (FastAPI) | CPU | ~100MB RAM | Query time |
| Qwen2.5:3b answer formatting | GPU | ~2-3GB VRAM | Query time |

**Peak GPU usage**: ~4GB VRAM at query time (Qwen2.5 + BGE-Small). Well within 8GB 4060 Max-Q.
**Ingestion**: ~1GB VRAM (BGE-Small batches only). CPU handles all parsing.

## Implementation Steps

### Step 1: Rewrite `orchestrator/ingest.py`

**Goal**: Zero LLM usage during ingestion. Two parallel pipelines.

#### Pipeline A — Structured Data Normalizer (CPU)

1. For each JSON file, find the element with `extracted_data`
2. Extract `reporting_period` / filename → determine quarter + fiscal year
3. Recursively flatten the nested dict:
   - `quarter_results.revenue_from_operations` → `revenue_from_operations` with period scope "quarter"
   - `nine_months_ended_dec_2024.profit_after_tax` → `profit_after_tax` with period scope "9M"
4. Strip unit suffixes from keys: `_crore` → unit="Cr", `_lakhs` → unit="Lakh", `_rupees` → unit="INR"
5. Fuzzy-match keys to standardized metric names via synonym dict:
   ```
   "revenue_from_operations" → "Revenue"
   "profit_after_tax"        → "Net Income"
   "basic_eps"               → "EPS"
   "finance_costs"           → "Interest Expense"
   "two_wheelers_sold_lakhs" → "Units Sold"
   ```
6. Create Neo4j nodes: `File → DataPoint → MetricDefinition`

#### Pipeline B — Text Chunking + Embedding (CPU parse, GPU embed)

1. Collect all non-empty `text` fields from each JSON file
2. Clean: strip HTML tags (BeautifulSoup), normalize whitespace
3. Chunk with token-aware splitting:
   - Target: ~300 tokens per chunk (~400 words, ~2000 chars)
   - Overlap: 15% (~45 tokens)
   - Respect sentence boundaries where possible
4. Enrich each chunk with metadata:
   - `source_file`: filename
   - `quarter`: Q1/Q2/Q3/Q4 (from filename)
   - `fiscal_year`: FY23/FY24/FY25 (from filename)
   - `chunk_index`: position in document
5. Batch embed with BGE-Small on GPU (batch size ~64, ~1GB VRAM)
6. Store in Neo4j: `File -[:HAS_CHUNK]-> Chunk` with embedding + metadata

### Step 2: Create `orchestrator/chat.py` (New File)

**FastAPI chatbot endpoint. LLM is the glue, not the processor.**

```
POST /chat
Body: {"message": "What was revenue in Q1 FY24?", "history": [...]}
Response: {"answer": "...", "sources": [...], "metrics_used": {...}}
```

#### Chat Pipeline:

1. **Intent Classification** (Qwen2.5:3b, single prompt, <100 token output):
   - Input: user message
   - Output JSON: `{intent, metrics_needed[], period, quant_endpoint}`
   - Intents: `lookup` | `calculate` | `compare` | `trend` | `general`

2. **Data Retrieval** (CPU, Neo4j):
   - Structured graph query first (exact match on MetricDefinition + period)
   - If not found → RAG fallback:
     - Embed query with BGE-Small (GPU, ~50ms)
     - Vector similarity search on Chunk nodes
     - Return top-3 chunks as context

3. **Calculation Routing** (CPU, Quant Engine):
   - Map intent → quant engine endpoint
   - Build payload from retrieved metrics
   - Call quant engine, get deterministic result

4. **Answer Formatting** (Qwen2.5:3b, single prompt):
   - Input: user question + retrieved data + calculation result
   - Output: natural language answer with cited values
   - Constraint: no invented numbers, only repeat what was retrieved

### Step 3: Update Supporting Files

- `orchestrator/requirements.txt` — Add `beautifulsoup4`
- `docker-compose.yml` — Expose chat service port (8501), add chat entrypoint
- `orchestrator/db_diagnostics.py` — Update to verify new data structure

### Step 4: Update `orchestrator/query.py`

- Refactor `generate_report()` to use the same data retrieval functions as `chat.py`
- Keep as a batch reporting tool (non-interactive)
- Share `fetch_metric()` and `rag_fallback()` as common utilities

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `orchestrator/ingest.py` | Rewrite | Smart normalizer + GPU-batched embeddings |
| `orchestrator/chat.py` | New | FastAPI chatbot endpoint |
| `orchestrator/query.py` | Update | Use shared retrieval utilities |
| `orchestrator/requirements.txt` | Update | Add beautifulsoup4 |
| `docker-compose.yml` | Update | Expose chat port |
| `quant_engine/main.py` | No change | Deterministic math stays as-is |
| `Modelfile` | No change | Qwen2.5:3b config stays as-is |

## Verification Checklist

1. **Ingestion completes without GPU spike**: `docker exec hero-orchestrator python ingest.py` — GPU usage should stay under 1.5GB VRAM
2. **All 15 quarters ingested**: Run `db_diagnostics.py` — should show DataPoints for Q1 FY23 through Q4 FY25
3. **Metric lookup works**: Test `fetch_metric("Revenue", "Q1 FY23")` returns 8392.54
4. **Vector search works**: Query "revenue Q1 2022" returns relevant text chunks
5. **Chat endpoint works**:
   - Lookup: `"What was Hero MotoCorp revenue in Q1 FY23?"` → `₹8,392.54 Cr`
   - Calculate: `"Calculate net margin for Q3 FY25"` → deterministic result from quant engine
   - Compare: `"Compare revenue Q1 FY24 vs Q1 FY23"` → growth calculation via quant engine
6. **Math is deterministic**: Same question always gives same numerical answer (Qwen temp=0, seed=42)
