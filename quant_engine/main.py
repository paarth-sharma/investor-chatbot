from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import math

app = FastAPI(
    title="Hero MotoCorp Quant Engine",
    description="Financial calculations and analytics engine",
    version="2.0.0"
)

# ============================================================================
# REQUEST MODELS
# ============================================================================

class GrowthRequest(BaseModel):
    current: float
    previous: float

class MarginRequest(BaseModel):
    numerator: float
    denominator: float

class DividendRequest(BaseModel):
    face_value: float
    percentage: float

class CAGRRequest(BaseModel):
    """Compound Annual Growth Rate"""
    beginning_value: float
    ending_value: float
    years: float

class ROERequest(BaseModel):
    """Return on Equity"""
    net_income: float
    shareholders_equity: float

class ROARequest(BaseModel):
    """Return on Assets"""
    net_income: float
    total_assets: float

class ROCERequest(BaseModel):
    """Return on Capital Employed"""
    ebit: float
    capital_employed: float

class EBITDAMarginRequest(BaseModel):
    ebitda: float
    revenue: float

class OperatingMarginRequest(BaseModel):
    operating_income: float
    revenue: float

class NetMarginRequest(BaseModel):
    net_income: float
    revenue: float

class AssetTurnoverRequest(BaseModel):
    revenue: float
    average_total_assets: float

class InventoryTurnoverRequest(BaseModel):
    cost_of_goods_sold: float
    average_inventory: float

class CurrentRatioRequest(BaseModel):
    current_assets: float
    current_liabilities: float

class DebtToEquityRequest(BaseModel):
    total_debt: float
    shareholders_equity: float

class InterestCoverageRequest(BaseModel):
    ebit: float
    interest_expense: float

class EPSRequest(BaseModel):
    """Earnings Per Share"""
    net_income: float
    shares_outstanding: float

class PERatioRequest(BaseModel):
    """Price to Earnings Ratio"""
    market_price_per_share: float
    earnings_per_share: float

class BookValueRequest(BaseModel):
    total_equity: float
    shares_outstanding: float

class WorkingCapitalRequest(BaseModel):
    current_assets: float
    current_liabilities: float

class CashConversionCycleRequest(BaseModel):
    """Days Sales Outstanding + Days Inventory Outstanding - Days Payable Outstanding"""
    days_sales_outstanding: float
    days_inventory_outstanding: float
    days_payable_outstanding: float

class BreakEvenRequest(BaseModel):
    fixed_costs: float
    price_per_unit: float
    variable_cost_per_unit: float

class QuickRatioRequest(BaseModel):
    """Quick Ratio (Acid Test)"""
    current_assets: float
    inventory: float
    current_liabilities: float

class GrossProfitMarginRequest(BaseModel):
    revenue: float
    cost_of_goods_sold: float

class EVRequest(BaseModel):
    """Enterprise Value"""
    market_cap: float
    total_debt: float
    cash_and_equivalents: float

class TaxRateRequest(BaseModel):
    tax_expense: float
    profit_before_tax: float

class RevenuePerUnitRequest(BaseModel):
    total_revenue: float
    units_sold: float

class SequentialGrowthRequest(BaseModel):
    """Quarter-over-Quarter or Month-over-Month growth"""
    current_period: float
    previous_period: float

class YoYGrowthRequest(BaseModel):
    """Year-over-Year growth"""
    current_year: float
    previous_year: float

class AverageRequest(BaseModel):
    """Calculate average of a list of values"""
    values: List[float]

class VarianceRequest(BaseModel):
    """Calculate variance"""
    values: List[float]

class StandardDeviationRequest(BaseModel):
    """Calculate standard deviation"""
    values: List[float]

# ============================================================================
# BASIC CALCULATIONS
# ============================================================================

@app.post("/growth")
def calculate_growth(req: GrowthRequest):
    """
    Generic growth calculation: ((Current - Previous) / Previous) * 100
    Used for: Revenue growth, profit growth, etc.
    """
    if req.previous == 0:
        return {"result": None, "unit": "%", "error": "Previous value is 0, cannot calculate growth"}
    
    val = ((req.current - req.previous) / req.previous) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/margin")
def calculate_margin(req: MarginRequest):
    """
    Generic margin/ratio: (Numerator / Denominator) * 100
    Used for: Any percentage calculation
    """
    if req.denominator == 0:
        return {"result": None, "unit": "%", "error": "Denominator is 0"}
    
    val = (req.numerator / req.denominator) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/dividend")
def calculate_dividend(req: DividendRequest):
    """Dividend per share: Face Value * (Percentage / 100)"""
    val = req.face_value * (req.percentage / 100)
    return {"result": round(val, 2), "unit": "INR"}

@app.post("/average")
def calculate_average(req: AverageRequest):
    """Calculate simple average"""
    if not req.values:
        return {"result": None, "error": "No values provided"}
    
    avg = sum(req.values) / len(req.values)
    return {"result": round(avg, 4), "unit": ""}

# ============================================================================
# GROWTH & TREND CALCULATIONS
# ============================================================================

@app.post("/cagr")
def calculate_cagr(req: CAGRRequest):
    """
    Compound Annual Growth Rate
    CAGR = ((Ending Value / Beginning Value) ^ (1 / Years)) - 1
    """
    if req.beginning_value <= 0:
        return {"result": None, "unit": "%", "error": "Beginning value must be positive"}
    if req.years <= 0:
        return {"result": None, "unit": "%", "error": "Years must be positive"}
    
    val = (math.pow(req.ending_value / req.beginning_value, 1 / req.years) - 1) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/sequential-growth")
def calculate_sequential_growth(req: SequentialGrowthRequest):
    """Quarter-over-Quarter or Month-over-Month growth"""
    return calculate_growth(GrowthRequest(current=req.current_period, previous=req.previous_period))

@app.post("/yoy-growth")
def calculate_yoy_growth(req: YoYGrowthRequest):
    """Year-over-Year growth"""
    return calculate_growth(GrowthRequest(current=req.current_year, previous=req.previous_year))

# ============================================================================
# PROFITABILITY RATIOS
# ============================================================================

@app.post("/roe")
def calculate_roe(req: ROERequest):
    """
    Return on Equity: (Net Income / Shareholders' Equity) * 100
    Measures profitability from shareholders' perspective
    """
    if req.shareholders_equity == 0:
        return {"result": None, "unit": "%", "error": "Shareholders equity is 0"}
    
    val = (req.net_income / req.shareholders_equity) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/roa")
def calculate_roa(req: ROARequest):
    """
    Return on Assets: (Net Income / Total Assets) * 100
    Measures how efficiently assets generate profit
    """
    if req.total_assets == 0:
        return {"result": None, "unit": "%", "error": "Total assets is 0"}
    
    val = (req.net_income / req.total_assets) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/roce")
def calculate_roce(req: ROCERequest):
    """
    Return on Capital Employed: (EBIT / Capital Employed) * 100
    Measures return on capital invested in the business
    """
    if req.capital_employed == 0:
        return {"result": None, "unit": "%", "error": "Capital employed is 0"}
    
    val = (req.ebit / req.capital_employed) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/gross-margin")
def calculate_gross_margin(req: GrossProfitMarginRequest):
    """
    Gross Profit Margin: ((Revenue - COGS) / Revenue) * 100
    """
    if req.revenue == 0:
        return {"result": None, "unit": "%", "error": "Revenue is 0"}
    
    gross_profit = req.revenue - req.cost_of_goods_sold
    val = (gross_profit / req.revenue) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/operating-margin")
def calculate_operating_margin(req: OperatingMarginRequest):
    """
    Operating Margin: (Operating Income / Revenue) * 100
    Shows profitability from core operations
    """
    if req.revenue == 0:
        return {"result": None, "unit": "%", "error": "Revenue is 0"}
    
    val = (req.operating_income / req.revenue) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/net-margin")
def calculate_net_margin(req: NetMarginRequest):
    """
    Net Profit Margin: (Net Income / Revenue) * 100
    Overall profitability after all expenses
    """
    if req.revenue == 0:
        return {"result": None, "unit": "%", "error": "Revenue is 0"}
    
    val = (req.net_income / req.revenue) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/ebitda-margin")
def calculate_ebitda_margin(req: EBITDAMarginRequest):
    """
    EBITDA Margin: (EBITDA / Revenue) * 100
    Operating profitability before depreciation and amortization
    """
    if req.revenue == 0:
        return {"result": None, "unit": "%", "error": "Revenue is 0"}
    
    val = (req.ebitda / req.revenue) * 100
    return {"result": round(val, 4), "unit": "%"}

# ============================================================================
# EFFICIENCY RATIOS
# ============================================================================

@app.post("/asset-turnover")
def calculate_asset_turnover(req: AssetTurnoverRequest):
    """
    Asset Turnover Ratio: Revenue / Average Total Assets
    Measures how efficiently assets generate revenue
    """
    if req.average_total_assets == 0:
        return {"result": None, "unit": "x", "error": "Average total assets is 0"}
    
    val = req.revenue / req.average_total_assets
    return {"result": round(val, 4), "unit": "x"}

@app.post("/inventory-turnover")
def calculate_inventory_turnover(req: InventoryTurnoverRequest):
    """
    Inventory Turnover: COGS / Average Inventory
    Shows how many times inventory is sold and replaced
    """
    if req.average_inventory == 0:
        return {"result": None, "unit": "x", "error": "Average inventory is 0"}
    
    val = req.cost_of_goods_sold / req.average_inventory
    return {"result": round(val, 4), "unit": "x"}

@app.post("/revenue-per-unit")
def calculate_revenue_per_unit(req: RevenuePerUnitRequest):
    """
    Revenue Per Unit: Total Revenue / Units Sold
    Average selling price per unit
    """
    if req.units_sold == 0:
        return {"result": None, "unit": "INR/unit", "error": "Units sold is 0"}
    
    val = req.total_revenue / req.units_sold
    return {"result": round(val, 2), "unit": "INR (Crores) per Lakh units"}

# ============================================================================
# LIQUIDITY RATIOS
# ============================================================================

@app.post("/current-ratio")
def calculate_current_ratio(req: CurrentRatioRequest):
    """
    Current Ratio: Current Assets / Current Liabilities
    Measures ability to pay short-term obligations
    Good ratio: > 1.5
    """
    if req.current_liabilities == 0:
        return {"result": None, "unit": "x", "error": "Current liabilities is 0"}
    
    val = req.current_assets / req.current_liabilities
    return {"result": round(val, 4), "unit": "x"}

@app.post("/quick-ratio")
def calculate_quick_ratio(req: QuickRatioRequest):
    """
    Quick Ratio (Acid Test): (Current Assets - Inventory) / Current Liabilities
    More conservative than current ratio
    Good ratio: > 1.0
    """
    if req.current_liabilities == 0:
        return {"result": None, "unit": "x", "error": "Current liabilities is 0"}
    
    val = (req.current_assets - req.inventory) / req.current_liabilities
    return {"result": round(val, 4), "unit": "x"}

@app.post("/working-capital")
def calculate_working_capital(req: WorkingCapitalRequest):
    """
    Working Capital: Current Assets - Current Liabilities
    Shows liquidity available for operations
    """
    val = req.current_assets - req.current_liabilities
    return {"result": round(val, 2), "unit": "INR (Crores)"}

# ============================================================================
# LEVERAGE RATIOS
# ============================================================================

@app.post("/debt-to-equity")
def calculate_debt_to_equity(req: DebtToEquityRequest):
    """
    Debt-to-Equity Ratio: Total Debt / Shareholders' Equity
    Measures financial leverage
    Lower is generally better
    """
    if req.shareholders_equity == 0:
        return {"result": None, "unit": "x", "error": "Shareholders equity is 0"}
    
    val = req.total_debt / req.shareholders_equity
    return {"result": round(val, 4), "unit": "x"}

@app.post("/interest-coverage")
def calculate_interest_coverage(req: InterestCoverageRequest):
    """
    Interest Coverage Ratio: EBIT / Interest Expense
    Measures ability to pay interest on debt
    Higher is better (typically > 3)
    """
    if req.interest_expense == 0:
        return {"result": None, "unit": "x", "error": "Interest expense is 0"}
    
    val = req.ebit / req.interest_expense
    return {"result": round(val, 4), "unit": "x"}

# ============================================================================
# VALUATION METRICS
# ============================================================================

@app.post("/eps")
def calculate_eps(req: EPSRequest):
    """
    Earnings Per Share: Net Income / Shares Outstanding
    Profit allocated to each share
    """
    if req.shares_outstanding == 0:
        return {"result": None, "unit": "INR", "error": "Shares outstanding is 0"}
    
    val = req.net_income / req.shares_outstanding
    return {"result": round(val, 2), "unit": "INR"}

@app.post("/pe-ratio")
def calculate_pe_ratio(req: PERatioRequest):
    """
    Price-to-Earnings Ratio: Market Price per Share / EPS
    Shows how much investors pay per rupee of earnings
    """
    if req.earnings_per_share == 0:
        return {"result": None, "unit": "x", "error": "EPS is 0"}
    
    val = req.market_price_per_share / req.earnings_per_share
    return {"result": round(val, 4), "unit": "x"}

@app.post("/book-value")
def calculate_book_value(req: BookValueRequest):
    """
    Book Value Per Share: Total Equity / Shares Outstanding
    Net asset value per share
    """
    if req.shares_outstanding == 0:
        return {"result": None, "unit": "INR", "error": "Shares outstanding is 0"}
    
    val = req.total_equity / req.shares_outstanding
    return {"result": round(val, 2), "unit": "INR"}

@app.post("/enterprise-value")
def calculate_enterprise_value(req: EVRequest):
    """
    Enterprise Value: Market Cap + Total Debt - Cash
    Total value of the company
    """
    val = req.market_cap + req.total_debt - req.cash_and_equivalents
    return {"result": round(val, 2), "unit": "INR (Crores)"}

# ============================================================================
# OTHER METRICS
# ============================================================================

@app.post("/tax-rate")
def calculate_tax_rate(req: TaxRateRequest):
    """
    Effective Tax Rate: (Tax Expense / Profit Before Tax) * 100
    """
    if req.profit_before_tax == 0:
        return {"result": None, "unit": "%", "error": "Profit before tax is 0"}
    
    val = (req.tax_expense / req.profit_before_tax) * 100
    return {"result": round(val, 4), "unit": "%"}

@app.post("/cash-conversion-cycle")
def calculate_cash_conversion_cycle(req: CashConversionCycleRequest):
    """
    Cash Conversion Cycle: DSO + DIO - DPO
    Time to convert inventory back to cash
    Lower is better
    """
    val = req.days_sales_outstanding + req.days_inventory_outstanding - req.days_payable_outstanding
    return {"result": round(val, 2), "unit": "days"}

@app.post("/break-even-units")
def calculate_break_even(req: BreakEvenRequest):
    """
    Break-Even Point: Fixed Costs / (Price - Variable Cost per Unit)
    Units needed to break even
    """
    contribution_margin = req.price_per_unit - req.variable_cost_per_unit
    
    if contribution_margin <= 0:
        return {"result": None, "unit": "units", "error": "Price must be greater than variable cost"}
    
    val = req.fixed_costs / contribution_margin
    return {"result": round(val, 2), "unit": "units"}

@app.post("/variance")
def calculate_variance(req: VarianceRequest):
    """Statistical variance of a dataset"""
    if len(req.values) < 2:
        return {"result": None, "error": "Need at least 2 values"}
    
    mean = sum(req.values) / len(req.values)
    variance = sum((x - mean) ** 2 for x in req.values) / len(req.values)
    return {"result": round(variance, 4), "unit": ""}

@app.post("/std-dev")
def calculate_std_dev(req: StandardDeviationRequest):
    """Standard deviation of a dataset"""
    if len(req.values) < 2:
        return {"result": None, "error": "Need at least 2 values"}
    
    mean = sum(req.values) / len(req.values)
    variance = sum((x - mean) ** 2 for x in req.values) / len(req.values)
    std_dev = math.sqrt(variance)
    return {"result": round(std_dev, 4), "unit": ""}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Hero MotoCorp Quant Engine",
        "version": "2.0.0",
        "available_endpoints": [
            "/growth", "/margin", "/dividend", "/cagr",
            "/roe", "/roa", "/roce",
            "/gross-margin", "/operating-margin", "/net-margin", "/ebitda-margin",
            "/asset-turnover", "/inventory-turnover", "/revenue-per-unit",
            "/current-ratio", "/quick-ratio", "/working-capital",
            "/debt-to-equity", "/interest-coverage",
            "/eps", "/pe-ratio", "/book-value", "/enterprise-value",
            "/tax-rate", "/cash-conversion-cycle", "/break-even-units",
            "/variance", "/std-dev", "/average"
        ]
    }

@app.get("/docs")
def get_docs():
    """Redirect to interactive API docs"""
    return {"message": "Visit /docs for interactive API documentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)