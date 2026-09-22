# Test Code with default inputs for the Portfolio Optimizer API. Inputs are the example test cases provided.
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

#This is for testing the health checkup included in the app
def test_root_and_health_endpoints():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

#This is for testing the endpoint that returns the list of optimization strategies that are defined in the app.
def test_list_strategies():
    res = client.get("/api/v1/strategies")
    assert res.status_code == 200
    strategies = res.json()
    assert len(strategies) >= 5
    strat_map = {s["id"]: s for s in strategies}
    assert strat_map["equal_weights"]["status"] == "active"
    assert strat_map["risk_parity"]["status"] == "active"
    assert strat_map["minimize_drawdown"]["status"] == "active"
    assert strat_map["minimize_volatility"]["status"] == "active"
    assert strat_map["maximize_sharpe_ratio"]["status"] == "active"

# For testing equal weights strategy as in the examples
def test_equal_weights_two_assets_scenario_1():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {
                "ticker": "IEFA",
                "security_name": "iShares Core MSCI EAFE ETF",
                "allocation": 25.0,
            },
            {
                "ticker": "SPY",
                "security_name": "State Street SPDR S&P 500 ETF Trust",
                "allocation": 75.0,
            },
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["optimization_strategy"] == "Equal weights"
    changes = data["allocation_changes"]
    assert len(changes) == 2

    # Check IEFA
    iefa = next(c for c in changes if c["ticker"] == "IEFA")
    assert iefa["current_weight"] == 25.0
    assert iefa["optimized_weight"] == 50.0
    assert iefa["change"] == 25.0

    # Check SPY
    spy = next(c for c in changes if c["ticker"] == "SPY")
    assert spy["current_weight"] == 75.0
    assert spy["optimized_weight"] == 50.0
    assert spy["change"] == -25.0

    # Check sum of optimized weights
    total_opt = sum(c["optimized_weight"] for c in changes)
    assert total_opt == 100.0

# A generic test for equal weights strategy with 5 entries 
def test_equal_weights_five_assets():
    payload = {
        "optimization_strategy": "equal_weights",
        "securities": [
            {"ticker": "IEFA", "security_name": "iShares Core MSCI EAFE ETF", "allocation": 20.0},
            {"ticker": "GLD", "security_name": "SPDR Gold Shares", "allocation": 20.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 20.0},
            {"ticker": "VEA", "security_name": "Vanguard Developed Markets Index Fund;ETF", "allocation": 20.0},
            {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 20.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    for sec in data["allocation_changes"]:
        assert sec["optimized_weight"] == 20.0
        assert sec["change"] == 0.0

# A generic test for equal weights strategy with 3 random entries 
def test_equal_weights_three_assets_exact_sum():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {"ticker": "AAA", "security_name": "Stock A", "allocation": 50.0},
            {"ticker": "BBB", "security_name": "Stock B", "allocation": 30.0},
            {"ticker": "CCC", "security_name": "Stock C", "allocation": 20.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = response.json()["allocation_changes"]
    total_opt = sum(c["optimized_weight"] for c in changes)
    assert round(total_opt, 2) == 100.0

# This is for testing that app correctly checks if the sum of allocations is 100% or not
def test_validation_allocation_sum_not_100():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 25.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 25.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "Sum of allocations across all securities must equal 100%" in response.text


# This is for testing that app correctly checks if the securities list is empty or not
def test_validation_empty_securities():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422


# This is for testing that app correctly checks for duplicate tickers
def test_validation_duplicate_tickers():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY 1", "allocation": 50.0},
            {"ticker": "SPY", "security_name": "SPY 2", "allocation": 50.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "Duplicate ticker(s)" in response.text

# This is for testing that app correctly checks if the min_weight is greater than max_weight
def test_validation_min_exceeds_max_weight():
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0, "min_weight": 60.0, "max_weight": 40.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 50.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "cannot exceed max_weight" in response.text

#This is for testing the risk parity strategy
def test_risk_parity_scenario_2():
    """Test Scenario 2 from assignment: VEA: 25%, AGG: 75% with Risk Parity."""
    payload = {
        "optimization_strategy": "Risk Parity",
        "securities": [
            {"ticker": "VEA", "security_name": "Vanguard Developed Markets Index Fund;ETF", "allocation": 25.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 75.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}
    
    assert abs(changes["VEA"] - 20.12) <= 0.1
    assert abs(changes["AGG"] - 79.88) <= 0.1


# This is for testing the minimize drawdown strategy
def test_minimize_drawdown_multi_asset():
    """Verify minimize drawdown strategy generates valid weights summing to 100%."""
    payload = {
        "optimization_strategy": "Minimize Drawdown",
        "securities": [
            {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 60.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 30.0},
            {"ticker": "GLD", "security_name": "SPDR Gold Shares", "allocation": 10.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = data["allocation_changes"]
    assert len(changes) == 3

    total_w = sum(c["optimized_weight"] for c in changes)
    assert round(total_w, 2) == 100.0

    # Low-drawdown assets like AGG should receive significant allocation
    agg = next(c for c in changes if c["ticker"] == "AGG")
    assert agg["optimized_weight"] > 40.0


def test_minimize_drawdown_respects_constraints():
    """Verify minimize drawdown strictly respects min_weight and max_weight bounds."""
    payload = {
        "optimization_strategy": "Minimize Drawdown",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0, "min_weight": 20.0, "max_weight": 40.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 50.0, "min_weight": 60.0, "max_weight": 80.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert 20.0 <= changes["SPY"] <= 40.0
    assert 60.0 <= changes["AGG"] <= 80.0
    assert round(changes["SPY"] + changes["AGG"], 2) == 100.0


# This is for testing the minimize volatility strategy (Scenario 3)
def test_minimize_volatility_scenario_3():
    """Test Scenario 3 from assignment: SPY: 60%, AGG: 30%, GLD: 10% with Minimize Volatility."""
    payload = {
        "optimization_strategy": "Minimize Volatility",
        "securities": [
            {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 60.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 30.0},
            {"ticker": "GLD", "security_name": "SPDR Gold Shares", "allocation": 10.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert round(sum(changes.values()), 2) == 100.0
    # Global minimum variance allocation strongly favors AGG with small SPY and GLD portions
    assert abs(changes["AGG"] - 91.21) <= 0.5
    assert abs(changes["SPY"] - 6.93) <= 0.5
    assert abs(changes["GLD"] - 1.86) <= 0.5


def test_minimize_volatility_respects_constraints():
    """Verify minimize volatility strictly respects min_weight and max_weight bounds."""
    payload = {
        "optimization_strategy": "Minimize Volatility",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0, "min_weight": 20.0, "max_weight": 40.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 50.0, "min_weight": 60.0, "max_weight": 80.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert 20.0 <= changes["SPY"] <= 40.0
    assert 60.0 <= changes["AGG"] <= 80.0
    assert round(changes["SPY"] + changes["AGG"], 2) == 100.0


# This is for testing the Maximize Sharpe Ratio strategy (Scenario 4: Unconstrained)
def test_maximize_sharpe_scenario_4_unconstrained():
    """Test Scenario 4: Maximize Sharpe Ratio with default historical Treasury Rf.
    Matches Finominal reference platform: SPY: 69.40%, GLD: 30.60%, others 0.00%.
    """
    payload = {
        "optimization_strategy": "Maximize Sharpe Ratio",
        "securities": [
            {"ticker": "IEFA", "security_name": "iShares Core MSCI EAFE ETF", "allocation": 20.0},
            {"ticker": "GLD", "security_name": "SPDR Gold Shares", "allocation": 20.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 20.0},
            {"ticker": "VEA", "security_name": "Vanguard Developed Markets Index Fund;ETF", "allocation": 20.0},
            {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust", "allocation": 20.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert round(sum(changes.values()), 2) == 100.0
    # With Rf = 0.02 (2.0%), SPY is 70.35% and GLD is 29.65%, within 1% to 1.5% of both Finominal benchmarks
    assert abs(changes["SPY"] - 69.40) <= 1.5
    assert abs(changes["GLD"] - 30.60) <= 1.5
    assert changes["AGG"] == 0.0
    assert changes["VEA"] == 0.0
    assert changes["IEFA"] == 0.0

