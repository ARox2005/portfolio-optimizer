# Portfolio Optimizer REST API

A basic REST API built with **Python** and **FastAPI** that replicates the core optimization engine behind Finominal's Portfolio Optimizer tool.

---

## Features

- **Equal Weights Optimization**: Assigns equal allocation to all input securities ($w = 100/N\%$).
- **Allocation Delta Tracking**:
  - `change`: Percentage point delta (`optimized_weight - current_weight`), exactly matching the Finominal specification.
  - `pct_change`: Relative percentage change (`((optimized - current) / current) * 100`).
- **Comprehensive Validation**:
  - Validates that current allocations sum to $100\%$ within a $\pm 0.01\%$ floating-point tolerance.
  - Validates ticker uniqueness and rejects empty portfolios.
  - Enforces constraint feasibility (detects if `min_weight` or `max_weight` conflicts with equal allocation).
- **Extensible Strategy Pattern Architecture**:
  - Implements a central `StrategyRegistry` and abstract `BaseOptimizerStrategy`.
  - Roadmap strategies (*Risk Parity*, *Minimize Volatility*, *Maximize Sharpe Ratio*, *Minimize Drawdown*, *Optimize Factor Exposure*) can be plugged in without changing endpoint contracts or controller logic.
- **Interactive Documentation**: Built-in Swagger UI and ReDoc.
- **Automated Test Suite**: Full test coverage with `pytest` and `TestClient`.

---

## Project Structure

```
FinominalAssignment/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application initialization & routes
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                    # Application configuration & constants
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── portfolio.py                 # Pydantic models & validation schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── optimizer/
│   │       ├── __init__.py
│   │       ├── base.py                  # Abstract base strategy
│   │       ├── registry.py              # Central strategy registry & roadmap
│   │       └── strategies/
│   │           ├── __init__.py
│   │           └── equal_weight.py      # Equal Weights strategy implementation
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           └── endpoints/
│               ├── __init__.py
│               └── optimizer.py         # REST API endpoints (/optimize, /strategies)
├── data/
│   ├── Backend Engineer Assignment.docx
│   └── Data.xlsx                        # Dataset containing Fund Info & Returns
├── tests/
│   ├── __init__.py
│   └── test_optimizer.py                # Unit & API integration tests
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Python 3.13 supported)

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the API Server

Start the FastAPI development server with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once started, the API will be available at:
- **API Root**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
---

## API Endpoints Reference

### 1. Optimize Portfolio
`POST /api/v1/optimize`

Calculates optimized portfolio weights and allocation deltas based on the specified strategy.

#### Request Body Example (Equal Weights - Scenario 1):
```json
{
  "optimization_strategy": "Equal weights",
  "securities": [
    {
      "ticker": "IEFA",
      "security_name": "iShares Core MSCI EAFE ETF",
      "allocation": 25.0
    },
    {
      "ticker": "SPY",
      "security_name": "State Street SPDR S&P 500 ETF Trust",
      "allocation": 75.0
    }
  ]
}
```

#### Response Example (200 OK):
```json
{
  "optimization_strategy": "Equal weights",
  "allocation_changes": [
    {
      "ticker": "IEFA",
      "security_name": "iShares Core MSCI EAFE ETF",
      "current_weight": 25.0,
      "optimized_weight": 50.0,
      "change": 25.0,
      "pct_change": 100.0
    },
    {
      "ticker": "SPY",
      "security_name": "State Street SPDR S&P 500 ETF Trust",
      "current_weight": 75.0,
      "optimized_weight": 50.0,
      "change": -25.0,
      "pct_change": -33.33
    }
  ],
  "factor_betas": null,
  "metadata": {
    "strategy_id": "equal_weights",
    "asset_count": 2,
    "formula": "w = 100 / N %",
    "total_optimized_weight": 100.0
  }
}
```

### 2. List Supported Strategies
`GET /api/v1/strategies`

Returns all optimization strategies registered in the system along with their operational status (`active` or `planned`).

#### Response Example:
```json
[
  {
    "id": "equal_weights",
    "name": "Equal weights",
    "description": "Assign equal allocation to all securities (w = 100/N%).",
    "status": "active"
  },
  {
    "id": "risk_parity",
    "name": "Risk Parity",
    "description": "Allocate weights such that each security contributes equally to total portfolio risk.",
    "status": "planned"
  },
  {
    "id": "minimize_volatility",
    "name": "Minimize Volatility",
    "description": "Find the allocation that produces the lowest portfolio volatility.",
    "status": "planned"
  }
]
```

### 3. Health Checks
- `GET /`: Service information, version, and documentation links.
- `GET /health`: Simple probe returning `{"status": "ok"}`.

---

## Testing with cURL

### Scenario 1 (IEFA 25% / SPY 75% -> Equal Weights):
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_strategy": "Equal weights",
    "securities": [
      {"ticker": "IEFA", "security_name": "iShares Core MSCI EAFE ETF", "allocation": 25.0},
      {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 75.0}
    ]
  }'
```

### 5-Asset Portfolio:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_strategy": "Equal weights",
    "securities": [
      {"ticker": "IEFA", "security_name": "iShares Core MSCI EAFE ETF", "allocation": 20.0},
      {"ticker": "GLD", "security_name": "SPDR Gold Shares", "allocation": 20.0},
      {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 20.0},
      {"ticker": "VEA", "security_name": "Vanguard Developed Markets Index Fund;ETF", "allocation": 20.0},
      {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 20.0}
    ]
  }'
```

---

## Running Automated Tests

Run the complete test suite using pytest inside the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

All unit and API integration tests run against edge cases (allocation summing, duplicate tickers, empty portfolios, constraint infeasibility, and exact rounding reconciliation).
