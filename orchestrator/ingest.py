import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# --- Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize BGE-Small on GPU (falls back to CPU if no CUDA) ---
print("[-] Loading BGE-Small embedding model...")
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
try:
    embedder = embedder.to('cuda')
    EMBED_DEVICE = 'cuda'
    print("[+] BGE-Small loaded on GPU")
except Exception:
    EMBED_DEVICE = 'cpu'
    print("[+] BGE-Small loaded on CPU (no CUDA available)")

# --- Neo4j Connection ---
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# --- Chunking Config ---
CHUNK_SIZE_CHARS = 2000     # ~300 tokens
CHUNK_OVERLAP_CHARS = 300   # ~15% overlap
EMBED_BATCH_SIZE = 64

# =============================================================================
# FILENAME → PERIOD PARSER
# =============================================================================
# Filename pattern: Q1-apr-jun-2022.json → Q1 FY23 (Indian fiscal year Apr-Mar)
FILENAME_PATTERN = re.compile(
    r'Q(\d)-(\w+)-(\w+)-(\d{4})\.json'
)

# Indian FY: Q1 Apr-Jun 2022 → FY 2022-23 → "FY23" (ending year)
# Q4 Jan-Mar 2025 → FY 2024-25 → "FY25"
QUARTER_TO_FY_OFFSET = {
    '1': 1,  # Q1 Apr-Jun YYYY → FY ends YYYY+1
    '2': 1,  # Q2 Jul-Sep YYYY → FY ends YYYY+1
    '3': 1,  # Q3 Oct-Dec YYYY → FY ends YYYY+1
    '4': 0,  # Q4 Jan-Mar YYYY → FY ends YYYY
}

def parse_filename_period(filename: str) -> Dict[str, str]:
    """Extract quarter and fiscal year from filename."""
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {"quarter": "Unknown", "fiscal_year": "Unknown"}
    q_num, _, _, year_str = m.groups()
    year = int(year_str)
    fy_end = year + QUARTER_TO_FY_OFFSET[q_num]
    return {
        "quarter": f"Q{q_num}",
        "fiscal_year": f"FY{str(fy_end)[-2:]}",
        "period_label": f"Q{q_num} FY{str(fy_end)[-2:]}",
    }


# =============================================================================
# SYNONYM MAP: extracted_data key fragments → Standardized Metric Name
# =============================================================================
# Order matters: first match wins. More specific patterns first.
METRIC_SYNONYMS: List[Tuple[List[str], str]] = [
    # Units / Volume
    (["two_wheelers_sold", "units_sold", "volume", "sales_volume"], "Units Sold"),
    # Revenue
    (["revenue_from_operations", "revenue_from_ops"], "Revenue"),
    (["annual_revenue_from_operations"], "Revenue"),
    (["other_income"], "Other Income"),
    (["total_income"], "Total Income"),
    # Expenses
    (["cost_of_materials_consumed", "cost_of_raw_materials"], "Cost of Goods Sold"),
    (["purchase_of_stock_in_trade"], "Purchase of Stock in Trade"),
    (["changes_in_inventories"], "Changes in Inventories"),
    (["employee_benefits_expense", "employee_benefit"], "Employee Benefits Expense"),
    (["finance_costs", "finance_cost", "interest_expense"], "Interest Expense"),
    (["depreciation_and_amortisation", "depreciation"], "Depreciation"),
    (["other_expenses"], "Other Expenses"),
    (["total_expenses"], "Total Expenses"),
    # Profit
    (["profit_before_tax_before_exceptional"], "Pre-Tax Income (Before Exceptional)"),
    (["exceptional_item_amount", "exceptional_item"], "Exceptional Items"),
    (["profit_before_tax_after_exceptional"], "Pre-Tax Income"),
    (["profit_before_tax", "pbt"], "Pre-Tax Income"),
    (["profit_before_share_of_associates"], "Profit Before Associates"),
    (["share_of_profit_loss_from_associates", "share_of_associates_loss"], "Share of Associates P&L"),
    (["annual_profit_after_tax"], "Net Income"),
    (["profit_after_tax", "pat", "net_profit", "profit_for_the_period"], "Net Income"),
    (["other_comprehensive_income"], "Other Comprehensive Income"),
    (["total_comprehensive_income"], "Total Comprehensive Income"),
    # Tax
    (["total_tax_expense", "tax_expense"], "Tax Expense"),
    (["current_tax"], "Current Tax"),
    (["deferred_tax"], "Deferred Tax"),
    # EPS
    (["earnings_per_share_basic", "basic_eps", "eps_basic"], "EPS"),
    (["earnings_per_share_diluted", "diluted_eps", "eps_diluted"], "EPS Diluted"),
    # Equity
    (["paid_up_equity_share_capital", "paid_up_capital"], "Paid Up Capital"),
    (["other_equity"], "Other Equity"),
    (["total_equity", "shareholders_equity", "net_worth"], "Total Equity"),
    # Balance Sheet
    (["total_assets"], "Total Assets"),
    (["total_liabilities"], "Total Liabilities"),
    (["total_current_assets", "current_assets"], "Current Assets"),
    (["total_current_liabilities", "current_liabilities"], "Current Liabilities"),
    (["inventories", "inventory", "stock_in_trade"], "Inventory"),
    (["cash_and_cash_equivalents", "cash_and_bank", "cash"], "Cash"),
    (["total_debt", "total_borrowings"], "Total Debt"),
    (["equity_share_capital"], "Equity Share Capital"),
    # EBITDA (often in highlights, not always in results)
    (["ebitda"], "EBITDA"),
    (["ebitda_margin"], "EBITDA Margin"),
    (["ebitda_growth"], "EBITDA Growth"),
    # Dividend
    (["dividend_per_share", "final_dividend_per_share", "total_dividend_per_share"], "Dividend"),
    (["dividend_percentage", "final_dividend_percentage"], "Dividend Percentage"),
    # Growth highlights
    (["revenue_growth_percentage", "revenue_growth"], "Revenue Growth"),
    (["pat_growth_percentage", "pat_growth"], "PAT Growth"),
]

# Unit suffix stripping
UNIT_SUFFIXES = {
    "_crore": "Cr",
    "_crores": "Cr",
    "_cr": "Cr",
    "_lakhs": "Lakh",
    "_lakh": "Lakh",
    "_rupees": "INR",
    "_inr": "INR",
    "_percentage": "%",
    "_percent": "%",
    "_pct": "%",
}


def strip_unit_suffix(key: str) -> Tuple[str, str]:
    """Strip unit suffix from key and return (clean_key, unit)."""
    key_lower = key.lower()
    for suffix, unit in UNIT_SUFFIXES.items():
        if key_lower.endswith(suffix):
            return key[:-len(suffix)], unit
    return key, "Cr"  # Default unit for Indian financial data in Crores


def humanize_key(key: str) -> str:
    """Convert a snake_case key into a human-readable metric name.
    e.g. 'scooter_sales_current_quarter' → 'Scooter Sales Current Quarter'
    """
    # Strip common noise suffixes that don't add meaning
    noise = ["_current_quarter", "_previous_quarter", "_current_year", "_previous_year"]
    cleaned = key.lower()
    for n in noise:
        cleaned = cleaned.replace(n, "")
    # Also strip unit suffixes (already handled separately)
    for suffix in UNIT_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:len(cleaned) - len(suffix)]
    # Convert to title case
    return cleaned.replace("_", " ").strip().title()


def match_metric_name(key: str) -> str:
    """Match a flattened key to a standardized metric name.
    Falls back to a humanized version of the key — NEVER drops data.
    """
    key_lower = key.lower()
    for synonyms, standard_name in METRIC_SYNONYMS:
        for syn in synonyms:
            if syn in key_lower:
                return standard_name
    # Fallback: humanize the key so ALL numeric data gets ingested
    return humanize_key(key)


# =============================================================================
# PIPELINE A: STRUCTURED DATA NORMALIZER (CPU only)
# =============================================================================

def find_financial_sections(data: dict) -> Dict[str, dict]:
    """
    Find standalone and consolidated financial result sections regardless of
    where they are in the schema. Returns {'standalone': {...}, 'consolidated': {...}}.
    """
    sections = {"standalone": {}, "consolidated": {}}

    def _search(d: dict, path: str = ""):
        for k, v in d.items():
            k_lower = k.lower()
            if isinstance(v, dict):
                # Check if this key IS a financial results section
                if "standalone" in k_lower and ("financial" in k_lower or "result" in k_lower):
                    sections["standalone"] = v
                elif "consolidated" in k_lower and ("financial" in k_lower or "result" in k_lower):
                    sections["consolidated"] = v
                elif k_lower == "financial_results":
                    # Could contain standalone/consolidated as children
                    if "standalone" in v or "standalone_results" in v:
                        sections["standalone"] = v.get("standalone", v.get("standalone_results", {}))
                    if "consolidated" in v or "consolidated_results" in v:
                        sections["consolidated"] = v.get("consolidated", v.get("consolidated_results", {}))
                    # If no sub-keys match, treat the whole thing as standalone
                    if not sections["standalone"] and not sections["consolidated"]:
                        sections["standalone"] = v
                else:
                    _search(v, f"{path}.{k}")
    _search(data)

    # Fallback: look for standard keys directly
    for candidate in ["standalone_financial_results", "financial_results_standalone"]:
        if candidate in data and not sections["standalone"]:
            sections["standalone"] = data[candidate]
    for candidate in ["consolidated_financial_results", "financial_results_consolidated"]:
        if candidate in data and not sections["consolidated"]:
            sections["consolidated"] = data[candidate]

    return sections


def extract_period_scope(key: str) -> Optional[str]:
    """Detect if a nested key represents a specific period scope."""
    key_lower = key.lower()
    if "quarter" in key_lower:
        return "quarter"
    elif "half_year" in key_lower or "six_month" in key_lower:
        return "half_year"
    elif "nine_month" in key_lower:
        return "nine_months"
    elif "year_ended" in key_lower or "annual" in key_lower:
        return "annual"
    return None


def flatten_financial_data(section: dict, base_period: dict) -> List[Dict[str, Any]]:
    """
    Flatten a financial results section into a list of standardized data points.
    Handles both flat schemas and nested period-scoped schemas.
    """
    data_points = []

    def _process_leaf(key: str, value: Any, period_scope: str, context_path: str):
        """Process a single leaf key-value pair."""
        if not isinstance(value, (int, float)):
            return
        if isinstance(value, bool):
            return

        clean_key, unit = strip_unit_suffix(key)
        metric_name = match_metric_name(clean_key)

        # Build period label
        if period_scope == "quarter":
            period_label = base_period.get("period_label", "Unknown")
        elif period_scope == "half_year":
            q = base_period.get("quarter", "")
            fy = base_period.get("fiscal_year", "")
            period_label = f"H1 {fy}" if q in ("Q1", "Q2") else f"H2 {fy}"
        elif period_scope == "nine_months":
            fy = base_period.get("fiscal_year", "")
            period_label = f"9M {fy}"
        elif period_scope == "annual":
            fy = base_period.get("fiscal_year", "")
            period_label = f"FY {fy}" if not fy.startswith("FY") else fy
        else:
            period_label = base_period.get("period_label", "Unknown")

        # Handle special units
        if "lakh" in key.lower() and unit == "Cr":
            unit = "Lakh"

        data_points.append({
            "metric_name": metric_name,
            "original_key": key,
            "value": float(value),
            "unit": unit,
            "period": period_label,
            "period_scope": period_scope,
            "source_path": context_path,
        })

    def _walk(d: dict, current_scope: str, path: str):
        for k, v in d.items():
            if isinstance(v, dict):
                # Check if this dict represents a period scope
                detected_scope = extract_period_scope(k)
                if detected_scope:
                    _walk(v, detected_scope, f"{path}.{k}")
                elif k.lower() in ("earnings_per_share", "eps"):
                    # Nested EPS — flatten basic/diluted
                    for eps_k, eps_v in v.items():
                        if isinstance(eps_v, (int, float)):
                            if "basic" in eps_k.lower():
                                _process_leaf("basic_eps", eps_v, current_scope, f"{path}.{k}.{eps_k}")
                            elif "diluted" in eps_k.lower():
                                _process_leaf("diluted_eps", eps_v, current_scope, f"{path}.{k}.{eps_k}")
                else:
                    _walk(v, current_scope, f"{path}.{k}")
            else:
                _process_leaf(k, v, current_scope, f"{path}.{k}")

    _walk(section, "quarter", "")
    return data_points


def extract_extra_metrics(extracted_data: dict, base_period: dict) -> List[Dict[str, Any]]:
    """
    Recursively search ALL sections of extracted_data for financial metrics
    that weren't already captured by the standalone/consolidated parsers.
    Skips sections already handled (standalone_financial_results, consolidated_*, financial_results).
    """
    data_points = []
    period_label = base_period.get("period_label", "Unknown")

    # Sections already processed by find_financial_sections — skip these
    SKIP_SECTIONS = {
        "standalone_financial_results", "financial_results_standalone",
        "consolidated_financial_results", "financial_results_consolidated",
        "financial_results",
        # Non-financial metadata sections
        "auditor_information", "auditor_review", "audit_opinion",
        "signatory_details", "company_information", "meeting_details",
        "board_meeting_details", "stock_exchange_filings", "stock_exchange_filing",
        "regulatory_compliance", "regulatory_matters", "regulatory_notes",
        "regulatory_disclosures", "notes_and_disclosures",
        "reporting_period", "financial_period",
        "agm_details", "director_appointments", "auditor_appointments",
    }

    def _extract_from_section(d: dict, path: str):
        for k, v in d.items():
            full_path = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                # Check for period-scoped sub-dicts (quarter vs annual)
                scope = extract_period_scope(k)
                if scope and scope != "quarter":
                    # Skip non-quarter scopes in extra metrics to avoid duplication
                    continue
                _extract_from_section(v, full_path)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                clean_key, unit = strip_unit_suffix(k)
                metric_name = match_metric_name(clean_key)
                # Handle special units from key name
                if "lakh" in k.lower() and unit == "Cr":
                    unit = "Lakh"
                data_points.append({
                    "metric_name": metric_name,
                    "original_key": k,
                    "value": float(v),
                    "unit": unit,
                    "period": period_label,
                    "period_scope": "quarter",
                    "source_path": full_path,
                })

    for section_key, section_val in extracted_data.items():
        if section_key.lower() in SKIP_SECTIONS:
            continue
        if isinstance(section_val, dict):
            _extract_from_section(section_val, section_key)

    return data_points


# =============================================================================
# PIPELINE B: TEXT CHUNKING + EMBEDDING (CPU parse, GPU embed)
# =============================================================================

def clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if '<' in text and '>' in text:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(texts: List[str], chunk_size: int = CHUNK_SIZE_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict[str, Any]]:
    """
    Chunk a list of text elements into overlapping windows.
    Respects sentence boundaries where possible.
    """
    # Concatenate all text with element separators
    full_text = " ".join(t for t in texts if t.strip())
    if not full_text.strip():
        return []

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(full_text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(full_text):
            # Look for sentence end (. ! ?) near the chunk boundary
            search_window = full_text[max(end - 200, start):end]
            last_period = max(
                search_window.rfind('. '),
                search_window.rfind('.\n'),
                search_window.rfind('? '),
                search_window.rfind('! '),
            )
            if last_period > 0:
                end = max(end - 200, start) + last_period + 1

        chunk_text_content = full_text[start:end].strip()
        if chunk_text_content:
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text_content,
                "char_start": start,
                "char_end": end,
            })
            chunk_idx += 1

        start = end - overlap
        if start >= len(full_text):
            break

    return chunks


def batch_embed(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
    """Batch embed texts using BGE-Small on GPU."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # BGE-Small recommends prepending instruction for retrieval
        prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in batch]
        embeddings = embedder.encode(prefixed, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        if i % (batch_size * 4) == 0 and i > 0:
            logger.info(f"  Embedded {i}/{len(texts)} chunks...")
    return all_embeddings


# =============================================================================
# NEO4J INGESTION
# =============================================================================

def ingest_structured_data(session, filename: str, data_points: List[Dict], context_type: str):
    """Store structured financial data points in Neo4j knowledge graph."""
    for dp in data_points:
        session.run("""
            MATCH (f:File {name: $filename})
            MERGE (md:MetricDefinition {name: $standard_name})
            CREATE (dp:DataPoint {
                value: $value,
                unit: $unit,
                period: $period,
                period_scope: $period_scope,
                original_key: $original_key,
                context: $context,
                source_path: $source_path,
                confidence: 'high'
            })
            MERGE (f)-[:REPORTED]->(dp)
            MERGE (dp)-[:IS_METRIC]->(md)
        """,
        filename=filename,
        standard_name=dp["metric_name"],
        original_key=dp["original_key"],
        value=dp["value"],
        unit=dp["unit"],
        period=dp["period"],
        period_scope=dp["period_scope"],
        context=context_type,
        source_path=dp["source_path"])


def ingest_chunks(session, filename: str, chunks: List[Dict], embeddings: List[List[float]],
                  metadata: Dict[str, str]):
    """Store text chunks with embeddings in Neo4j for vector search."""
    for chunk, embedding in zip(chunks, embeddings):
        session.run("""
            MATCH (f:File {name: $filename})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.embedding = $embedding,
                c.chunk_index = $chunk_index,
                c.quarter = $quarter,
                c.fiscal_year = $fiscal_year,
                c.period_label = $period_label
            MERGE (f)-[:HAS_CHUNK]->(c)
        """,
        filename=filename,
        chunk_id=f"{filename}_c{chunk['chunk_index']}",
        text=chunk["text"],
        embedding=embedding,
        chunk_index=chunk["chunk_index"],
        quarter=metadata.get("quarter", ""),
        fiscal_year=metadata.get("fiscal_year", ""),
        period_label=metadata.get("period_label", ""))


# =============================================================================
# MAIN INGESTION PIPELINE
# =============================================================================

def ingest_file(filepath: str):
    """
    Ingest a single JSON file through both pipelines:
    Pipeline A: Structured data → Neo4j knowledge graph (CPU only)
    Pipeline B: Raw text → chunked, embedded, stored for RAG (CPU + GPU)
    """
    filename = os.path.basename(filepath)
    logger.info(f"Processing {filename}...")

    with open(filepath, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning(f"Unexpected format in {filename}: expected list, got {type(data)}")
        return

    # --- Parse period from filename ---
    period_info = parse_filename_period(filename)
    logger.info(f"  Period: {period_info['period_label']}")

    # --- Create File node ---
    with driver.session() as session:
        session.run("""
            MERGE (f:File {name: $filename})
            SET f.processed_at = datetime(),
                f.quarter = $quarter,
                f.fiscal_year = $fiscal_year,
                f.period_label = $period_label
        """, filename=filename, **period_info)

    # =========================================================================
    # PIPELINE A: Structured Data Extraction (CPU only, no LLM)
    # =========================================================================
    extracted_data_elem = None
    for elem in data:
        if elem.get("metadata", {}).get("extracted_data"):
            extracted_data_elem = elem["metadata"]["extracted_data"]
            break

    total_structured = 0
    if extracted_data_elem:
        sections = find_financial_sections(extracted_data_elem)

        with driver.session() as session:
            # Standalone results
            if sections["standalone"]:
                standalone_dps = flatten_financial_data(sections["standalone"], period_info)
                ingest_structured_data(session, filename, standalone_dps, "Standalone")
                total_structured += len(standalone_dps)
                logger.info(f"  Standalone: {len(standalone_dps)} data points")

            # Consolidated results
            if sections["consolidated"]:
                consolidated_dps = flatten_financial_data(sections["consolidated"], period_info)
                ingest_structured_data(session, filename, consolidated_dps, "Consolidated")
                total_structured += len(consolidated_dps)
                logger.info(f"  Consolidated: {len(consolidated_dps)} data points")

            # Extra metrics (highlights, dividends, equity, balance sheet)
            extra_dps = extract_extra_metrics(extracted_data_elem, period_info)
            if extra_dps:
                ingest_structured_data(session, filename, extra_dps, "Unknown")
                total_structured += len(extra_dps)
                logger.info(f"  Extra metrics: {len(extra_dps)} data points")

        logger.info(f"  [Pipeline A] Total structured data points: {total_structured}")
    else:
        logger.warning(f"  No extracted_data found in {filename}")

    # =========================================================================
    # PIPELINE B: Text Chunking + Embedding (CPU + GPU)
    # =========================================================================
    raw_texts = []
    for elem in data:
        text = elem.get("text", "").strip()
        if text:
            raw_texts.append(clean_html(text))

    if not raw_texts:
        logger.warning(f"  No text elements found in {filename}")
        return

    logger.info(f"  [Pipeline B] Processing {len(raw_texts)} text elements...")

    # Chunk
    chunks = chunk_text(raw_texts)
    logger.info(f"  Created {len(chunks)} chunks")

    if not chunks:
        return

    # Embed (GPU batch)
    chunk_texts = [c["text"] for c in chunks]
    logger.info(f"  Embedding {len(chunk_texts)} chunks on {EMBED_DEVICE}...")
    embeddings = batch_embed(chunk_texts)

    # Store in Neo4j
    with driver.session() as session:
        ingest_chunks(session, filename, chunks, embeddings, period_info)

    logger.info(f"  [Pipeline B] Stored {len(chunks)} chunks with embeddings")
    logger.info(f"  DONE: {filename} — {total_structured} metrics, {len(chunks)} chunks\n")


def main():
    """Main entry point: set up indexes and ingest all data files."""
    # --- Setup Neo4j indexes and constraints ---
    with driver.session() as session:
        try:
            session.run("""
                CREATE VECTOR INDEX financial_vectors IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("Vector index ready")
        except Exception as e:
            logger.info(f"Vector index setup note: {e}")

        try:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (m:MetricDefinition) REQUIRE m.name IS UNIQUE
            """)
            logger.info("MetricDefinition constraint ready")
        except Exception as e:
            logger.info(f"Constraint setup note: {e}")

    # --- Ingest all JSON files ---
    json_files = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.endswith('.json')
    ])

    logger.info(f"Found {len(json_files)} JSON files to ingest")
    for filepath in json_files:
        ingest_file(filepath)

    # --- Summary ---
    with driver.session() as session:
        counts = session.run("""
            MATCH (f:File) WITH count(f) as files
            MATCH (dp:DataPoint) WITH files, count(dp) as datapoints
            MATCH (c:Chunk) WITH files, datapoints, count(c) as chunks
            MATCH (md:MetricDefinition) WITH files, datapoints, chunks, count(md) as metrics
            RETURN files, datapoints, chunks, metrics
        """).single()
        logger.info(f"\n{'='*60}")
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"  Files:       {counts['files']}")
        logger.info(f"  DataPoints:  {counts['datapoints']}")
        logger.info(f"  Chunks:      {counts['chunks']}")
        logger.info(f"  Metric Defs: {counts['metrics']}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
