import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --- Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QUANT_URL = os.getenv("QUANT_ENGINE_URL", "http://localhost:8000")
OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = OLLAMA_BASE + "/api/chat"
# finma = Qwen2.5:3b fine-tuned for JSON extraction (system prompt forces JSON output)
# qwen2.5:3b = base model for natural language conversation
MODEL_EXTRACT = os.getenv("FINMA_MODEL", "finma")
MODEL_CHAT = "qwen2.5:3b"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

print("[-] Loading BGE-Small embedding model...")
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
try:
    embedder = embedder.to('cuda')
    print("[+] BGE-Small loaded on GPU")
except Exception:
    print("[+] BGE-Small loaded on CPU")

app = FastAPI(title="Hero MotoCorp Financial Chatbot", version="2.0.0")


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    answer: str
    data: Dict[str, Any] = {}
    calculations: Dict[str, Any] = {}
    sources: List[str] = []
    debug: Dict[str, Any] = {}


# =============================================================================
# METRIC ALIASES
# =============================================================================
METRIC_ALIASES: Dict[str, List[str]] = {
    "Revenue":              ["Revenue", "Revenue from Operations", "Total Income"],
    "Net Income":           ["Net Income", "Profit After Tax", "Net Profit"],
    "EBITDA":               ["EBITDA"],
    "EBIT":                 ["EBIT", "Operating Profit"],
    "Operating Income":     ["Operating Income", "Profit from Operations"],
    "Cost of Goods Sold":   ["Cost of Goods Sold"],
    "Interest Expense":     ["Interest Expense"],
    "Tax Expense":          ["Tax Expense"],
    "Pre-Tax Income":       ["Pre-Tax Income", "Pre-Tax Income (Before Exceptional)"],
    "Total Assets":         ["Total Assets"],
    "Total Equity":         ["Total Equity", "Other Equity"],
    "Total Debt":           ["Total Debt"],
    "Current Assets":       ["Current Assets"],
    "Current Liabilities":  ["Current Liabilities"],
    "Inventory":            ["Inventory"],
    "Cash":                 ["Cash"],
    "Shares Outstanding":   ["Shares Outstanding", "Paid Up Capital"],
    "EPS":                  ["EPS"],
    "EPS Diluted":          ["EPS Diluted"],
    "Dividend":             ["Dividend"],
    "Units Sold":           ["Units Sold"],
    "Total Expenses":       ["Total Expenses"],
    "Other Income":         ["Other Income"],
    "Depreciation":         ["Depreciation"],
    "Total Comprehensive Income": ["Total Comprehensive Income"],
}

ENDPOINT_REQUIREMENTS = {
    "net-margin":         {"needs": ["Net Income", "Revenue"], "map": {"net_income": "Net Income", "revenue": "Revenue"}},
    "ebitda-margin":      {"needs": ["EBITDA", "Revenue"], "map": {"ebitda": "EBITDA", "revenue": "Revenue"}},
    "operating-margin":   {"needs": ["Operating Income", "Revenue"], "map": {"operating_income": "Operating Income", "revenue": "Revenue"}},
    "gross-margin":       {"needs": ["Revenue", "Cost of Goods Sold"], "map": {"revenue": "Revenue", "cost_of_goods_sold": "Cost of Goods Sold"}},
    "roe":                {"needs": ["Net Income", "Total Equity"], "map": {"net_income": "Net Income", "shareholders_equity": "Total Equity"}},
    "roa":                {"needs": ["Net Income", "Total Assets"], "map": {"net_income": "Net Income", "total_assets": "Total Assets"}},
    "roce":               {"needs": ["EBIT", "Total Assets", "Current Liabilities"], "map": None},
    "current-ratio":      {"needs": ["Current Assets", "Current Liabilities"], "map": {"current_assets": "Current Assets", "current_liabilities": "Current Liabilities"}},
    "quick-ratio":        {"needs": ["Current Assets", "Inventory", "Current Liabilities"], "map": {"current_assets": "Current Assets", "inventory": "Inventory", "current_liabilities": "Current Liabilities"}},
    "working-capital":    {"needs": ["Current Assets", "Current Liabilities"], "map": {"current_assets": "Current Assets", "current_liabilities": "Current Liabilities"}},
    "debt-to-equity":     {"needs": ["Total Debt", "Total Equity"], "map": {"total_debt": "Total Debt", "shareholders_equity": "Total Equity"}},
    "interest-coverage":  {"needs": ["EBIT", "Interest Expense"], "map": {"ebit": "EBIT", "interest_expense": "Interest Expense"}},
    "eps":                {"needs": ["Net Income", "Shares Outstanding"], "map": {"net_income": "Net Income", "shares_outstanding": "Shares Outstanding"}},
    "book-value":         {"needs": ["Total Equity", "Shares Outstanding"], "map": {"total_equity": "Total Equity", "shares_outstanding": "Shares Outstanding"}},
    "tax-rate":           {"needs": ["Tax Expense", "Pre-Tax Income"], "map": {"tax_expense": "Tax Expense", "profit_before_tax": "Pre-Tax Income"}},
    "asset-turnover":     {"needs": ["Revenue", "Total Assets"], "map": {"revenue": "Revenue", "average_total_assets": "Total Assets"}},
    "inventory-turnover": {"needs": ["Cost of Goods Sold", "Inventory"], "map": {"cost_of_goods_sold": "Cost of Goods Sold", "average_inventory": "Inventory"}},
    "revenue-per-unit":   {"needs": ["Revenue", "Units Sold"], "map": {"total_revenue": "Revenue", "units_sold": "Units Sold"}},
}


# =============================================================================
# DATA RETRIEVAL
# =============================================================================

def get_available_periods() -> List[str]:
    with driver.session() as session:
        records = session.run("""
            MATCH (dp:DataPoint) WHERE dp.period_scope = 'quarter'
            RETURN DISTINCT dp.period AS period ORDER BY period
        """)
        return [r["period"] for r in records]


def get_available_metrics() -> List[str]:
    with driver.session() as session:
        records = session.run("MATCH (md:MetricDefinition) RETURN md.name AS name ORDER BY name")
        return [r["name"] for r in records]


def fetch_metric(metric_name: str, period: str) -> Optional[float]:
    """Retrieve a metric. Tries aliases, then fuzzy name match. Prefers Consolidated > Standalone."""
    aliases = METRIC_ALIASES.get(metric_name, [metric_name])
    for name in aliases:
        with driver.session() as session:
            result = session.run("""
                MATCH (md:MetricDefinition {name: $name})<-[:IS_METRIC]-(dp:DataPoint)
                WHERE dp.period = $period AND dp.period_scope = 'quarter'
                WITH dp ORDER BY
                    CASE dp.context WHEN 'Consolidated' THEN 1 WHEN 'Standalone' THEN 2 ELSE 3 END,
                    CASE dp.confidence WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
                RETURN dp.value AS value, dp.unit AS unit, dp.context AS context
                LIMIT 1
            """, name=name, period=period).single()
            if result:
                return result['value']

    # Fuzzy fallback: search for metric names containing the query terms
    with driver.session() as session:
        result = session.run("""
            MATCH (md:MetricDefinition)<-[:IS_METRIC]-(dp:DataPoint)
            WHERE dp.period = $period AND dp.period_scope = 'quarter'
              AND toLower(md.name) CONTAINS toLower($search)
            WITH dp, md ORDER BY
                CASE dp.context WHEN 'Consolidated' THEN 1 WHEN 'Standalone' THEN 2 ELSE 3 END
            RETURN dp.value AS value, dp.unit AS unit, md.name AS matched_name
            LIMIT 1
        """, search=metric_name, period=period).single()
        if result:
            logger.info(f"Fuzzy match: '{metric_name}' → '{result['matched_name']}'")
            return result['value']
    return None


def fetch_metric_all_periods(metric_name: str) -> Dict[str, float]:
    """Fetch a metric across ALL quarters. Returns {period: value}."""
    aliases = METRIC_ALIASES.get(metric_name, [metric_name])
    results = {}
    for name in aliases:
        with driver.session() as session:
            for record in session.run("""
                MATCH (md:MetricDefinition {name: $name})<-[:IS_METRIC]-(dp:DataPoint)
                WHERE dp.period_scope = 'quarter'
                WITH dp.period AS period, dp ORDER BY
                    CASE dp.context WHEN 'Consolidated' THEN 1 WHEN 'Standalone' THEN 2 ELSE 3 END
                WITH period, collect(dp)[0] AS best
                RETURN period, best.value AS value
            """, name=name):
                p = record["period"]
                if p not in results:
                    results[p] = record["value"]
    return results


def rag_search(query_text: str, top_k: int = 5, period_filter: str = None) -> List[Dict[str, Any]]:
    prefixed = f"Represent this sentence for searching relevant passages: {query_text}"
    query_embedding = embedder.encode(prefixed).tolist()
    cypher = "CALL db.index.vector.queryNodes('financial_vectors', $top_k, $embedding) YIELD node, score "
    if period_filter:
        cypher += "WHERE node.period_label = $period "
    cypher += "RETURN node.text AS text, node.id AS chunk_id, node.period_label AS period_label, score ORDER BY score DESC"
    params = {"top_k": top_k, "embedding": query_embedding}
    if period_filter:
        params["period"] = period_filter
    results = []
    with driver.session() as session:
        for record in session.run(cypher, **params):
            results.append({"text": record["text"], "chunk_id": record["chunk_id"],
                            "period_label": record["period_label"], "score": record["score"]})
    return results


def call_quant_engine(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{QUANT_URL}/{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# LLM CALLS
# =============================================================================

def preprocess_calendar_to_fy(message: str) -> str:
    """Convert calendar year references to Indian FY format in the message.
    e.g., 'Q3 of 2022' → 'Q3 FY23', 'Q1 2024' → 'Q1 FY25'
    Does this in Python so the 3b LLM doesn't have to do arithmetic.
    """
    import re
    # Match patterns like "Q3 of 2022", "Q3 2022", "Q1-2024", "q2 of calendar year 2023"
    pattern = r'[Qq](\d)\s*(?:of\s*(?:calendar\s*(?:year\s*)?)?)?(\d{4})'

    def _replace(m):
        q = int(m.group(1))
        year = int(m.group(2))
        # Indian FY: Q1-Q3 (Apr-Dec) → FY ends year+1; Q4 (Jan-Mar) → FY ends same year
        fy_end = year + 1 if q in (1, 2, 3) else year
        return f"Q{q} FY{str(fy_end)[-2:]}"

    return re.sub(pattern, _replace, message)


def llm_parse_query(user_message: str) -> Dict[str, Any]:
    """
    Use finma (JSON extraction model) to decompose question into structured query plan.
    Given the EXACT available periods and metrics — cannot hallucinate names.
    """
    # Pre-convert any calendar year references to FY format
    processed_message = preprocess_calendar_to_fy(user_message)
    if processed_message != user_message:
        logger.info(f"Calendar→FY conversion: '{user_message}' → '{processed_message}'")

    available_periods = get_available_periods()
    available_metrics = get_available_metrics()

    prompt = f"""You are a query planner for Hero MotoCorp financial data (Indian two-wheeler company).

Decompose this question into a structured query plan.

QUESTION: "{processed_message}"

AVAILABLE PERIODS (use ONLY these exact strings):
{json.dumps(available_periods)}

AVAILABLE METRICS (use ONLY these exact strings):
{json.dumps(available_metrics)}

NOTE: Calendar years have already been converted to Indian FY format in the QUESTION above.
The periods in the question are ready to match against AVAILABLE PERIODS directly.

COLLOQUIAL → METRIC NAME EXAMPLES:
- "scooters/bikes/vehicles/two wheelers sold", "units", "volume" → "Units Sold"
- "profit", "earnings", "PAT", "bottom line" → "Net Income"
- "sales", "top line" → "Revenue"
- "tax" → "Tax Expense"

IMPORTANT — GRANULAR METRICS:
- The AVAILABLE METRICS list includes granular breakdowns (e.g., scooter-specific, motorcycle-specific, segment-wise).
- If the user asks about a SPECIFIC sub-category (scooters, motorcycles, exports, etc.), find the CLOSEST matching metric name from the list.
- If the user asks about the OVERALL/TOTAL, prefer the general metric (e.g., "Units Sold" over "Scooter Sales").
- Always prefer an EXACT match from the available metrics list over a guess.

QUERY TYPES:
- "lookup": fetch metric(s) for period(s)
- "compare": same metric, 2 periods, compute growth
- "trend": same metric, 3+ periods
- "calculate": financial ratio (set quant_endpoint)
- "general": anything else

CALCULATIONS: net-margin, ebitda-margin, operating-margin, gross-margin, roe, roa, roce, current-ratio, quick-ratio, working-capital, debt-to-equity, interest-coverage, eps, book-value, tax-rate, asset-turnover, inventory-turnover, revenue-per-unit

Return JSON with these fields:
- query_type: one of "lookup", "compare", "trend", "calculate", "general"
- metrics: list of metric names from AVAILABLE METRICS
- periods: list of period strings from AVAILABLE PERIODS
- quant_endpoint: null or one of the CALCULATIONS endpoints
- reasoning: brief explanation of your interpretation

RULES:
1. periods and metrics MUST come from the AVAILABLE lists above. Nothing else.
2. For lookup: 1+ metrics, 1+ periods. Simple data retrieval.
3. For compare: exactly 2 periods. FIRST = newer, SECOND = older.
4. For trend: all periods in chronological order.
5. "between 2022 and 2024" for Q3 means Q3 FY23 (Oct-Dec 2022) and Q3 FY25 (Oct-Dec 2024).
6. "last 2 years" means current vs 2 FY earlier.
7. If the question asks about a SINGLE period, use "lookup" NOT "compare".
8. Read the QUESTION carefully — extract the exact periods and metrics mentioned."""

    payload = {
        "model": MODEL_EXTRACT,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096}
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()['message']['content']
        parsed = json.loads(content)
        logger.info(f"Query plan: {json.dumps(parsed, indent=2)}")
        return parsed
    except Exception as e:
        logger.error(f"Query parsing failed: {e}")
        return {"query_type": "general", "metrics": [], "periods": [], "quant_endpoint": None, "reasoning": str(e)}


def llm_format_answer(user_message: str, retrieved_data: Dict[str, Any],
                       calculations: Dict[str, Any] = None,
                       rag_context: List[str] = None) -> str:
    """
    Use qwen2.5:3b BASE model (not finma) to format a natural language English answer.
    finma outputs JSON — we need conversational English here.
    """
    parts = [f"USER QUESTION: {user_message}\n"]

    if retrieved_data:
        parts.append("VERIFIED DATA FROM DATABASE:")
        for key, value in retrieved_data.items():
            if value is not None:
                parts.append(f"  {key} = {value}")
            else:
                parts.append(f"  {key} = DATA NOT AVAILABLE")

    if calculations:
        parts.append("\nCALCULATION RESULTS:")
        for label, result in calculations.items():
            if isinstance(result, dict) and "result" in result:
                parts.append(f"  {label} = {result['result']} {result.get('unit', '')}")
            elif isinstance(result, dict) and "error" in result:
                parts.append(f"  {label} = Could not compute: {result['error']}")
            else:
                parts.append(f"  {label} = {result}")

    if rag_context:
        parts.append("\nSUPPORTING TEXT FROM FINANCIAL REPORTS:")
        for i, text in enumerate(rag_context[:3], 1):
            parts.append(f"  [{i}] {text[:400]}")

    context = "\n".join(parts)

    prompt = f"""Answer the user's financial question in clear, professional English sentences.

RULES:
- Use ONLY the numbers from the VERIFIED DATA and CALCULATION RESULTS above. Do not make up any numbers.
- Write in plain English sentences, NOT JSON or bullet points.
- If data is not available, say so clearly — state what IS available and what is NOT.
- If the user asks about a SPECIFIC sub-category (e.g., "scooters only") but the data only has an aggregate (e.g., "motorcycles and scooters combined"), explicitly say the breakdown is not available and provide the aggregate instead. NEVER attribute an aggregate number to a sub-category.
- For growth comparisons: state both values, the absolute change, and the growth percentage.
- Use Indian number format: Crores (Cr), Lakhs where appropriate.
- Keep the answer concise: 2-4 sentences maximum.
- Reference the SUPPORTING TEXT to add relevant qualitative context (product launches, market share notes, etc.).

{context}

Answer in English:"""

    payload = {
        "model": MODEL_CHAT,  # Use base qwen2.5:3b for natural language
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 2048}
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()['message']['content']
    except Exception as e:
        logger.error(f"Answer formatting failed: {e}")
        # Deterministic fallback without LLM
        lines = []
        for key, value in retrieved_data.items():
            if value is not None:
                lines.append(f"{key}: {value}")
        if calculations:
            for label, result in calculations.items():
                if isinstance(result, dict) and "result" in result:
                    lines.append(f"{label}: {result['result']} {result.get('unit', '')}")
        return ". ".join(lines) if lines else "Unable to retrieve data for your query."


# =============================================================================
# QUERY PROCESSORS
# =============================================================================

def process_lookup(plan: Dict, user_message: str) -> ChatResponse:
    metrics = plan.get("metrics", [])
    periods = plan.get("periods", [])
    if not metrics:
        return ChatResponse(answer="I couldn't determine which metric you're asking about. "
                                    "Try specifying: revenue, profit, EPS, units sold, etc.")
    if not periods:
        return ChatResponse(answer="Please specify a period, for example: Q1 FY24, Q3 FY25.")

    retrieved = {}
    sources = []
    for period in periods:
        for metric in metrics:
            value = fetch_metric(metric, period)
            key = f"{metric} ({period})"
            retrieved[key] = value
            if value is not None:
                sources.append(f"graph:{metric}@{period}")

    # Always run RAG to provide textual context — helps the LLM give nuanced,
    # specific answers (e.g., scooter vs motorcycle breakdowns from raw text)
    rag_query = f"{user_message} {' '.join(metrics)} {' '.join(periods)}"
    chunks = rag_search(rag_query, top_k=5,
                         period_filter=periods[0] if len(periods) == 1 else None)
    rag_texts = [c["text"] for c in chunks]
    sources.extend([f"rag:{c['chunk_id']}" for c in chunks])

    answer = llm_format_answer(user_message, retrieved, rag_context=rag_texts)
    return ChatResponse(answer=answer, data=retrieved, sources=sources,
                         debug={"query_type": "lookup", "plan": plan})


def process_compare(plan: Dict, user_message: str) -> ChatResponse:
    metrics = plan.get("metrics", [])
    periods = plan.get("periods", [])
    if not metrics:
        return ChatResponse(answer="Which metric should I compare? Try: revenue, profit, units sold, etc.")
    if len(periods) < 2:
        return ChatResponse(answer="I need two periods to compare, for example: Q3 FY25 vs Q3 FY23.")

    metric = metrics[0]
    p_new, p_old = periods[0], periods[1]
    val_new = fetch_metric(metric, p_new)
    val_old = fetch_metric(metric, p_old)

    retrieved = {f"{metric} ({p_new})": val_new, f"{metric} ({p_old})": val_old}
    calculations = {}
    sources = []

    if val_new is not None and val_old is not None:
        diff = round(val_new - val_old, 2)
        calculations["Absolute Change"] = {"result": diff, "unit": ""}
        growth_result = call_quant_engine("growth", {"current": val_new, "previous": val_old})
        calculations["Growth Percentage"] = growth_result
        sources = [f"graph:{metric}@{p_new}", f"graph:{metric}@{p_old}", "quant:growth"]
    else:
        missing = [p for p, v in [(p_new, val_new), (p_old, val_old)] if v is None]
        for p in missing:
            chunks = rag_search(f"{metric} {p}", top_k=2, period_filter=p)
            sources.extend([f"rag:{c['chunk_id']}" for c in chunks])

    # Always include RAG context for richer answers
    rag_chunks = rag_search(f"{user_message} {metric} {p_new} {p_old}", top_k=4)
    rag_texts = [c["text"] for c in rag_chunks]
    sources.extend([f"rag:{c['chunk_id']}" for c in rag_chunks])

    answer = llm_format_answer(user_message, retrieved, calculations, rag_context=rag_texts)
    return ChatResponse(answer=answer, data=retrieved, calculations=calculations,
                         sources=sources, debug={"query_type": "compare", "plan": plan})


def process_trend(plan: Dict, user_message: str) -> ChatResponse:
    metrics = plan.get("metrics", [])
    periods = plan.get("periods", [])
    if not metrics:
        return ChatResponse(answer="Which metric should I show the trend for?")

    metric = metrics[0]
    if not periods:
        all_data = fetch_metric_all_periods(metric)
        periods = sorted(all_data.keys())
    else:
        all_data = {}
        for p in periods:
            val = fetch_metric(metric, p)
            if val is not None:
                all_data[p] = val

    if not all_data:
        return ChatResponse(answer=f"No data found for {metric}.")

    retrieved = {f"{metric} ({p})": all_data.get(p) for p in periods if p in all_data}
    calculations = {}
    sorted_periods = sorted(all_data.keys())
    for i in range(1, len(sorted_periods)):
        p_prev, p_curr = sorted_periods[i - 1], sorted_periods[i]
        growth = call_quant_engine("growth", {"current": all_data[p_curr], "previous": all_data[p_prev]})
        calculations[f"Growth {p_prev} → {p_curr}"] = growth

    answer = llm_format_answer(user_message, retrieved, calculations)
    return ChatResponse(answer=answer, data=retrieved, calculations=calculations,
                         sources=[f"graph:{metric}@{p}" for p in sorted_periods],
                         debug={"query_type": "trend", "plan": plan})


def process_calculate(plan: Dict, user_message: str) -> ChatResponse:
    endpoint = plan.get("quant_endpoint")
    periods = plan.get("periods", [])
    if not endpoint or endpoint not in ENDPOINT_REQUIREMENTS:
        return ChatResponse(answer="I couldn't determine which calculation you need. "
                                    "Try: net margin, ROE, current ratio, EPS, etc.")
    if not periods:
        return ChatResponse(answer="Please specify a period. Example: Q1 FY25.")

    period = periods[0]
    req = ENDPOINT_REQUIREMENTS[endpoint]
    retrieved = {}
    missing = []
    for metric in req["needs"]:
        value = fetch_metric(metric, period)
        retrieved[f"{metric} ({period})"] = value
        if value is None:
            missing.append(metric)

    if missing:
        rag_texts = []
        for m in missing:
            chunks = rag_search(f"{m} {period}", top_k=2, period_filter=period)
            rag_texts.extend([c["text"] for c in chunks])
        answer = llm_format_answer(user_message, retrieved, rag_context=rag_texts)
        return ChatResponse(answer=f"Missing data for: {', '.join(missing)}. {answer}",
                             data=retrieved, debug={"query_type": "calculate", "plan": plan})

    raw_values = {m: fetch_metric(m, period) for m in req["needs"]}
    if endpoint == "roce":
        payload = {"ebit": raw_values["EBIT"],
                   "capital_employed": raw_values["Total Assets"] - raw_values["Current Liabilities"]}
    else:
        payload = {k: raw_values[v] for k, v in req["map"].items()}

    calc_result = call_quant_engine(endpoint, payload)
    calculations = {endpoint: calc_result}
    answer = llm_format_answer(user_message, retrieved, calculations)
    return ChatResponse(answer=answer, data=retrieved, calculations=calculations,
                         sources=[f"graph:{m}@{period}" for m in req["needs"]] + [f"quant:{endpoint}"],
                         debug={"query_type": "calculate", "plan": plan})


def process_general(plan: Dict, user_message: str) -> ChatResponse:
    chunks = rag_search(user_message, top_k=5)
    rag_texts = [c["text"] for c in chunks]
    answer = llm_format_answer(user_message, {}, rag_context=rag_texts)
    return ChatResponse(answer=answer, sources=[f"rag:{c['chunk_id']}" for c in chunks],
                         debug={"query_type": "general", "plan": plan})


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f"Chat: {req.message}")
    plan = llm_parse_query(req.message)
    query_type = plan.get("query_type", "general")

    if query_type == "lookup":
        return process_lookup(plan, req.message)
    elif query_type == "compare":
        return process_compare(plan, req.message)
    elif query_type == "trend":
        return process_trend(plan, req.message)
    elif query_type == "calculate":
        return process_calculate(plan, req.message)
    else:
        return process_general(plan, req.message)


@app.get("/")
def health():
    return {"status": "healthy", "service": "Hero MotoCorp Financial Chatbot", "version": "2.0.0"}


@app.get("/metrics")
def list_metrics():
    """All metrics with period coverage — use this to verify ingestion."""
    with driver.session() as session:
        records = session.run("""
            MATCH (md:MetricDefinition)<-[:IS_METRIC]-(dp:DataPoint)
            WHERE dp.period_scope = 'quarter'
            WITH md.name AS metric, collect(DISTINCT dp.period) AS periods, count(dp) AS count
            RETURN metric, periods, count ORDER BY metric
        """)
        return [dict(r) for r in records]


@app.get("/periods")
def list_periods():
    return get_available_periods()


@app.get("/debug/data/{period}")
def debug_data(period: str):
    """Show ALL data points for a given period — use to verify ingestion correctness."""
    with driver.session() as session:
        records = session.run("""
            MATCH (dp:DataPoint)-[:IS_METRIC]->(md:MetricDefinition)
            WHERE dp.period = $period AND dp.period_scope = 'quarter'
            RETURN md.name AS metric, dp.value AS value, dp.unit AS unit,
                   dp.context AS context, dp.original_key AS original_key,
                   dp.source_path AS source_path
            ORDER BY md.name
        """, period=period)
        return [dict(r) for r in records]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
