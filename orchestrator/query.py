import os
import json
import requests
from neo4j import GraphDatabase
from typing import Optional
from sentence_transformers import SentenceTransformer

# --- Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QUANT_URL = os.getenv("QUANT_ENGINE_URL", "http://localhost:8000")
OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = OLLAMA_BASE + "/api/chat"
MODEL_NAME = os.getenv("FINMA_MODEL", "finma")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# BGE-Small for RAG fallback
print("[-] Loading BGE-Small embedding model...")
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
try:
    embedder = embedder.to('cuda')
    print("[+] BGE-Small loaded on GPU")
except Exception:
    print("[+] BGE-Small loaded on CPU")


# --- Shared Retrieval Utilities ---
def fetch_metric(metric_name: str, period_keyword: str) -> Optional[float]:
    """
    Retrieve the most relevant value for a standardized metric.
    Prefers Consolidated > Standalone, high confidence first.
    Filters to quarter-scope data points only.
    """
    # Try common aliases for each metric
    ALIASES = {
        "Revenue": ["Revenue", "Revenue from Operations", "Total Income"],
        "NetIncome": ["Net Income", "Profit After Tax", "Net Profit"],
        "Net Income": ["Net Income", "Profit After Tax", "Net Profit"],
        "EBITDA": ["EBITDA"],
        "EBIT": ["EBIT", "Operating Profit"],
        "OperatingIncome": ["Operating Income", "Profit from Operations"],
        "Operating Income": ["Operating Income", "Profit from Operations"],
        "CostOfGoodsSold": ["Cost of Goods Sold"],
        "Cost of Goods Sold": ["Cost of Goods Sold"],
        "InterestExpense": ["Interest Expense"],
        "Interest Expense": ["Interest Expense"],
        "TaxExpense": ["Tax Expense"],
        "Tax Expense": ["Tax Expense"],
        "PreTaxIncome": ["Pre-Tax Income", "Pre-Tax Income (Before Exceptional)"],
        "Pre-Tax Income": ["Pre-Tax Income", "Pre-Tax Income (Before Exceptional)"],
        "TotalAssets": ["Total Assets"],
        "Total Assets": ["Total Assets"],
        "TotalEquity": ["Total Equity", "Other Equity"],
        "Total Equity": ["Total Equity", "Other Equity"],
        "TotalDebt": ["Total Debt"],
        "Total Debt": ["Total Debt"],
        "CurrentAssets": ["Current Assets"],
        "Current Assets": ["Current Assets"],
        "CurrentLiabilities": ["Current Liabilities"],
        "Current Liabilities": ["Current Liabilities"],
        "Inventory": ["Inventory"],
        "Cash": ["Cash"],
        "SharesOutstanding": ["Shares Outstanding", "Paid Up Capital"],
        "Shares Outstanding": ["Shares Outstanding", "Paid Up Capital"],
        "EPS": ["EPS"],
        "Dividend": ["Dividend"],
        "UnitsSold": ["Units Sold"],
        "Units Sold": ["Units Sold"],
    }

    name_variants = ALIASES.get(metric_name, [metric_name])

    for name in name_variants:
        query = """
        MATCH (md:MetricDefinition {name: $name})<-[:IS_METRIC]-(dp:DataPoint)
        WHERE toLower(dp.period) CONTAINS toLower($period)
          AND dp.period_scope = 'quarter'
        WITH dp
        ORDER BY
            CASE dp.context WHEN 'Consolidated' THEN 1 WHEN 'Standalone' THEN 2 ELSE 3 END,
            CASE dp.confidence WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
        RETURN dp.value as value
        LIMIT 1
        """
        with driver.session() as session:
            result = session.run(query, name=name, period=period_keyword).single()
            if result:
                return result['value']
    return None


def rag_fallback(metric_description: str, period_keyword: str) -> Optional[float]:
    """
    If a metric is not found in the structured graph, attempt to retrieve it via vector search.
    Embeds the query, searches chunks, and uses LLM to extract the specific value.
    """
    # Embed the search query
    search_text = f"{metric_description} {period_keyword}"
    prefixed = f"Represent this sentence for searching relevant passages: {search_text}"
    query_embedding = embedder.encode(prefixed).tolist()

    # Vector similarity search
    cypher = """
    CALL db.index.vector.queryNodes('financial_vectors', 3, $embedding)
    YIELD node, score
    WHERE toLower(node.period_label) CONTAINS toLower($period)
    RETURN node.text AS text, score
    ORDER BY score DESC
    """

    chunks = []
    with driver.session() as session:
        records = session.run(cypher, embedding=query_embedding, period=period_keyword)
        for record in records:
            chunks.append(record["text"])

    if not chunks:
        print(f"      RAG fallback: no relevant chunks found for '{metric_description}' in {period_keyword}")
        return None

    # Use LLM to extract the specific numeric value from the chunks
    context = "\n---\n".join(chunks[:3])
    prompt = f"""Extract the exact numeric value for "{metric_description}" from the text below.
The period is: {period_keyword}
Return ONLY a JSON object: {{"value": <number or null>}}
If the value is not found, return {{"value": null}}

TEXT:
{context}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 2048}
    }

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()['message']['content']
        parsed = json.loads(content)
        value = parsed.get("value")
        if value is not None:
            print(f"      RAG fallback found: {metric_description} = {value}")
            return float(value)
    except Exception as e:
        print(f"      RAG fallback error: {e}")

    return None


def call_quant_engine(endpoint: str, payload: dict) -> dict:
    """Send POST request to Quant Engine and return parsed JSON."""
    try:
        resp = requests.post(f"{QUANT_URL}/{endpoint}", json=payload, timeout=30)
        if resp.status_code == 422:
            return {"error": "Validation Error", "details": resp.json()}
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# --- Main Reporting Function ---
def generate_report(target_period: str):
    """
    Fetches all required financial metrics for a given period,
    calls the Quant Engine endpoints, and prints a comprehensive report.
    """
    print(f"\n{'='*60}")
    print(f" FINANCIAL REPORT - Period: {target_period}")
    print(f"{'='*60}\n")

    # ---- 1. Define all metrics needed by the Quant Engine ----
    METRICS_MAP = {
        "Revenue": "Revenue",
        "NetIncome": "Net Income",
        "EBITDA": "EBITDA",
        "EBIT": "EBIT",
        "OperatingIncome": "Operating Income",
        "CostOfGoodsSold": "Cost of Goods Sold",
        "InterestExpense": "Interest Expense",
        "TaxExpense": "Tax Expense",
        "PreTaxIncome": "Pre-Tax Income",
        "TotalAssets": "Total Assets",
        "TotalEquity": "Total Equity",
        "TotalDebt": "Total Debt",
        "CurrentAssets": "Current Assets",
        "CurrentLiabilities": "Current Liabilities",
        "Inventory": "Inventory",
        "Cash": "Cash",
        "SharesOutstanding": "Shares Outstanding",
        "EPS": "EPS",
        "Dividend": "Dividend",
        "UnitsSold": "Units Sold",
    }

    # ---- 2. Fetch all available metrics ----
    data = {}
    missing = []
    print(" Fetching structured data from knowledge graph...")
    for key, metric_name in METRICS_MAP.items():
        value = fetch_metric(metric_name, target_period)
        if value is None:
            # Attempt RAG fallback
            value = rag_fallback(metric_name, target_period)
        if value is not None:
            data[key] = value
            print(f"   {key:20} = {value:>15,.2f}")
        else:
            data[key] = None
            missing.append(key)
            print(f"   {key:20} - not found")

    if missing:
        print(f"\n  Warning: Missing metrics for {target_period}: {', '.join(missing)}")
        print("   Some calculations will be skipped.\n")

    # ---- 3. Call Quant Engine endpoints ----
    print(" Computing financial ratios and metrics...\n")
    results = {}

    # PROFITABILITY & MARGINS
    if data.get("Revenue") and data["Revenue"] > 0:
        rev = data["Revenue"]
        if data.get("NetIncome"):
            res = call_quant_engine("net-margin", {"net_income": data["NetIncome"], "revenue": rev})
            results["Net Margin"] = res.get("result", "N/A")
        if data.get("EBITDA"):
            res = call_quant_engine("ebitda-margin", {"ebitda": data["EBITDA"], "revenue": rev})
            results["EBITDA Margin"] = res.get("result", "N/A")
        if data.get("OperatingIncome"):
            res = call_quant_engine("operating-margin", {"operating_income": data["OperatingIncome"], "revenue": rev})
            results["Operating Margin"] = res.get("result", "N/A")
        if data.get("CostOfGoodsSold"):
            res = call_quant_engine("gross-margin", {"revenue": rev, "cost_of_goods_sold": data["CostOfGoodsSold"]})
            results["Gross Margin"] = res.get("result", "N/A")

    # RETURN RATIOS
    if data.get("NetIncome"):
        if data.get("TotalEquity") and data["TotalEquity"] > 0:
            res = call_quant_engine("roe", {"net_income": data["NetIncome"], "shareholders_equity": data["TotalEquity"]})
            results["ROE"] = res.get("result", "N/A")
        if data.get("TotalAssets") and data["TotalAssets"] > 0:
            res = call_quant_engine("roa", {"net_income": data["NetIncome"], "total_assets": data["TotalAssets"]})
            results["ROA"] = res.get("result", "N/A")

    if data.get("EBIT") and data.get("TotalAssets") and data.get("CurrentLiabilities"):
        capital_employed = data["TotalAssets"] - data["CurrentLiabilities"]
        if capital_employed > 0:
            res = call_quant_engine("roce", {"ebit": data["EBIT"], "capital_employed": capital_employed})
            results["ROCE"] = res.get("result", "N/A")

    # LIQUIDITY RATIOS
    if data.get("CurrentAssets") and data.get("CurrentLiabilities"):
        ca, cl = data["CurrentAssets"], data["CurrentLiabilities"]
        if cl > 0:
            res = call_quant_engine("current-ratio", {"current_assets": ca, "current_liabilities": cl})
            results["Current Ratio"] = res.get("result", "N/A")
        if data.get("Inventory") is not None:
            res = call_quant_engine("quick-ratio", {"current_assets": ca, "inventory": data["Inventory"], "current_liabilities": cl})
            results["Quick Ratio"] = res.get("result", "N/A")
        res = call_quant_engine("working-capital", {"current_assets": ca, "current_liabilities": cl})
        results["Working Capital"] = res.get("result", "N/A")

    # LEVERAGE RATIOS
    if data.get("TotalDebt") and data.get("TotalEquity") and data["TotalEquity"] > 0:
        res = call_quant_engine("debt-to-equity", {"total_debt": data["TotalDebt"], "shareholders_equity": data["TotalEquity"]})
        results["Debt/Equity"] = res.get("result", "N/A")
    if data.get("EBIT") and data.get("InterestExpense") and data["InterestExpense"] > 0:
        res = call_quant_engine("interest-coverage", {"ebit": data["EBIT"], "interest_expense": data["InterestExpense"]})
        results["Interest Coverage"] = res.get("result", "N/A")

    # EFFICIENCY RATIOS
    if data.get("Revenue") and data.get("TotalAssets") and data["TotalAssets"] > 0:
        res = call_quant_engine("asset-turnover", {"revenue": data["Revenue"], "average_total_assets": data["TotalAssets"]})
        results["Asset Turnover"] = res.get("result", "N/A")
    if data.get("CostOfGoodsSold") and data.get("Inventory") and data["Inventory"] > 0:
        res = call_quant_engine("inventory-turnover", {"cost_of_goods_sold": data["CostOfGoodsSold"], "average_inventory": data["Inventory"]})
        results["Inventory Turnover"] = res.get("result", "N/A")
    if data.get("Revenue") and data.get("UnitsSold") and data["UnitsSold"] > 0:
        res = call_quant_engine("revenue-per-unit", {"total_revenue": data["Revenue"], "units_sold": data["UnitsSold"]})
        results["Revenue Per Unit"] = res.get("result", "N/A")

    # VALUATION METRICS
    if data.get("NetIncome") and data.get("SharesOutstanding") and data["SharesOutstanding"] > 0:
        res = call_quant_engine("eps", {"net_income": data["NetIncome"], "shares_outstanding": data["SharesOutstanding"]})
        results["EPS"] = res.get("result", "N/A")
    if data.get("TotalEquity") and data.get("SharesOutstanding") and data["SharesOutstanding"] > 0:
        res = call_quant_engine("book-value", {"total_equity": data["TotalEquity"], "shares_outstanding": data["SharesOutstanding"]})
        results["Book Value per Share"] = res.get("result", "N/A")

    # OTHER METRICS
    if data.get("TaxExpense") and data.get("PreTaxIncome") and data["PreTaxIncome"] > 0:
        res = call_quant_engine("tax-rate", {"tax_expense": data["TaxExpense"], "profit_before_tax": data["PreTaxIncome"]})
        results["Effective Tax Rate"] = res.get("result", "N/A")

    if data.get("Dividend"):
        results["Dividend per Share"] = data["Dividend"]

    # ---- 4. Print report ----
    print(" QUANT ENGINE RESULTS")
    print("-" * 60)
    for metric, value in results.items():
        if value != "N/A" and value is not None:
            if isinstance(value, (int, float)):
                print(f"   {metric:25} : {value:>15,.2f}")
            else:
                print(f"   {metric:25} : {value}")
        else:
            print(f"   {metric:25} : N/A (insufficient data)")
    print("-" * 60)
    print(f"\n{'='*60}\n")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        period = sys.argv[1]
    else:
        period = input("Enter target period (e.g., 'Q3 FY24', 'Q1 FY25'): ")
    generate_report(period)
