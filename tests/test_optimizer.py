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
    assert len(strategies) >= 6
    strat_map = {s["id"]: s for s in strategies}
    assert strat_map["equal_weights"]["status"] == "active"
    assert strat_map["risk_parity"]["status"] == "active"
    assert strat_map["minimize_drawdown"]["status"] == "active"
    assert strat_map["minimize_volatility"]["status"] == "active"
    assert strat_map["maximize_sharpe_ratio"]["status"] == "active"
    assert strat_map["optimize_factor_exposure"]["status"] == "active"

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


# Tests for optional Minimum Dividend Yield constraint across strategies

def test_equal_weights_with_constraints_ignored():
    """Verify that passing optional portfolio constraints does not alter Equal Weights."""
    payload = {
        "optimization_strategy": "Equal weights",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 25.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 75.0},
        ],
        "constraints": {
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert changes["IEFA"] == 50.0
    assert changes["SPY"] == 50.0


def test_risk_parity_with_constraints_ignored():
    """Verify that passing optional portfolio constraints does not alter Risk Parity."""
    payload = {
        "optimization_strategy": "Risk Parity",
        "securities": [
            {"ticker": "VEA", "security_name": "VEA", "allocation": 25.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 75.0},
        ],
        "constraints": {
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert abs(changes["VEA"] - 20.12) <= 0.5
    assert abs(changes["AGG"] - 79.88) <= 0.5


def test_maximize_sharpe_scenario_5_constrained():
    """Test Scenario 5: Maximize Sharpe with min_weight 5%, max_weight 40%, min_dividend_yield 2.50%."""
    payload = {
        "optimization_strategy": "Maximize Sharpe Ratio",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "VEA", "security_name": "VEA", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
        ],
        "constraints": {
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert round(sum(changes.values()), 2) == 100.0
    for ticker, weight in changes.items():
        assert 5.0 <= weight <= 40.0

    # AGG should be at max cap (40%) to satisfy the 2.5% dividend yield requirement
    assert changes["AGG"] == 40.0
    assert abs(changes["SPY"] - 36.24) <= 1.0
    assert abs(changes["IEFA"] - 13.76) <= 1.0
    assert changes["GLD"] == 5.0
    assert changes["VEA"] == 5.0

    # Check achieved dividend yield: AGG 3.974%, IEFA 3.278%, VEA 2.034%, SPY 0.987%, GLD 0%
    achieved_yield = (
        changes["AGG"] * 0.03974
        + changes["IEFA"] * 0.03278
        + changes["VEA"] * 0.02034
        + changes["SPY"] * 0.00987
        + changes["GLD"] * 0.0
    )
    assert achieved_yield >= 2.49  # >= 2.50% subject to penny rounding


def test_minimize_volatility_with_dividend_yield():
    """Verify Minimize Volatility respects min_dividend_yield constraint."""
    payload = {
        "optimization_strategy": "Minimize Volatility",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 50.0},
        ],
        "constraints": {
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert round(sum(changes.values()), 2) == 100.0
    # AGG dividend yield is 3.97%, SPY is 0.99% -> requires AGG >= 50.5% to achieve 2.50%
    achieved_yield = (changes["AGG"] * 0.03974 + changes["SPY"] * 0.00987)
    assert achieved_yield >= 2.49


def test_minimize_drawdown_with_dividend_yield():
    """Verify Minimize Drawdown respects min_dividend_yield constraint."""
    payload = {
        "optimization_strategy": "Minimize Drawdown",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 50.0},
        ],
        "constraints": {
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert round(sum(changes.values()), 2) == 100.0
    achieved_yield = (changes["AGG"] * 0.03974 + changes["SPY"] * 0.00987)
    assert achieved_yield >= 2.49


def test_infeasible_dividend_yield_returns_422():
    """Verify that an unachievable dividend yield returns HTTP 422 error."""
    payload = {
        "optimization_strategy": "Maximize Sharpe Ratio",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 50.0},
        ],
        "constraints": {
            "min_dividend_yield": 10.0  # Impossible since SPY is ~0.99% and GLD is 0%
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "infeasible" in response.json()["detail"].lower()


def test_optimize_factor_exposure_scenario_6_maximize_momentum():
    """Test Scenario 6: Optimize Factor Exposure maximizing Momentum exposure."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "IEFA", "security_name": "iShares Core MSCI EAFE ETF", "allocation": 20.0},
            {"ticker": "GLD", "security_name": "SPDR Gold Trust", "allocation": 20.0},
            {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF", "allocation": 20.0},
            {"ticker": "VEA", "security_name": "Vanguard FTSE Developed Markets ETF", "allocation": 20.0},
            {"ticker": "SPY", "security_name": "SPDR S&P 500 ETF Trust", "allocation": 20.0},
        ],
        "constraints": {
            "momentum": "maximize"
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["optimization_strategy"] == "Optimize Factor Exposure"
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    assert round(sum(changes.values()), 2) == 100.0
    # VEA has highest momentum beta (0.187) among the 5 assets, so unconstrained it receives 100%
    assert changes["VEA"] == 100.0
    assert changes["SPY"] == 0.0
    assert changes["IEFA"] == 0.0
    assert changes["GLD"] == 0.0
    assert changes["AGG"] == 0.0


def test_optimize_factor_exposure_momentum_with_bounds_and_dividend():
    """Test Scenario 6 with security min/max bounds and minimum dividend yield constraint."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "VEA", "security_name": "VEA", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 20.0, "min_weight": 5.0, "max_weight": 40.0},
        ],
        "constraints": {
            "momentum": "maximize",
            "min_dividend_yield": 2.50
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    changes = {c["ticker"]: c["optimized_weight"] for c in data["allocation_changes"]}

    # Verify budget and bounds
    assert round(sum(changes.values()), 2) == 100.0
    for ticker, weight in changes.items():
        assert 5.0 <= weight <= 40.0

    # Highest momentum & dividend combination (VEA and IEFA) at max weight 40%
    assert changes["VEA"] == 40.0
    assert changes["IEFA"] == 40.0
    assert changes["GLD"] == 5.0

    # Verify achieved dividend yield is >= 2.50%
    achieved_yield = (
        changes["AGG"] * 0.0344
        + changes["IEFA"] * 0.0312
        + changes["VEA"] * 0.0291
        + changes["SPY"] * 0.0125
        + changes["GLD"] * 0.0
    )
    assert achieved_yield >= 2.49


def test_optimize_factor_exposure_nested_factors_schema():
    """Verify that constraints under nested 'factors' key work identically."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 20.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 20.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 20.0},
            {"ticker": "VEA", "security_name": "VEA", "allocation": 20.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 20.0},
        ],
        "constraints": {
            "factors": {
                "momentum": "maximize"
            }
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert changes["VEA"] == 100.0


def test_optimize_factor_exposure_value_and_size():
    """Verify multi-factor optimization targeting value maximization and size minimization."""
    payload = {
        "optimization_strategy": "optimize_factor_exposure",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 20.0, "min_weight": 10.0, "max_weight": 50.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 20.0, "min_weight": 10.0, "max_weight": 50.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 20.0, "min_weight": 10.0, "max_weight": 50.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 20.0, "min_weight": 10.0, "max_weight": 50.0},
            {"ticker": "VEA", "security_name": "VEA", "allocation": 20.0, "min_weight": 10.0, "max_weight": 50.0},
        ],
        "constraints": {
            "value": "maximize",
            "size": "minimize"
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    changes = {c["ticker"]: c["optimized_weight"] for c in response.json()["allocation_changes"]}
    assert round(sum(changes.values()), 2) == 100.0
    for w in changes.values():
        assert 10.0 <= w <= 50.0


def test_optimize_factor_exposure_infeasible_dividend():
    """Verify that an unachievable dividend yield raises HTTP 422."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 50.0},
        ],
        "constraints": {
            "momentum": "maximize",
            "min_dividend_yield": 10.0
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "infeasible" in response.json()["detail"].lower()


def test_optimize_factor_exposure_infeasible_bounds():
    """Verify that conflicting security bounds (sum(min_weight) > 100%) raise HTTP 422."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "SPY", "security_name": "SPY", "allocation": 50.0, "min_weight": 60.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 50.0, "min_weight": 60.0},
        ],
        "constraints": {
            "momentum": "maximize"
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "infeasible" in response.json()["detail"].lower()

@pytest.mark.parametrize(
    "strategy_name",
    [
        "Equal weights",
        "Risk Parity",
        "Minimize Volatility",
        "Minimize Drawdown",
        "Maximize Sharpe Ratio",
        "Optimize Factor Exposure",
    ]
)
def test_factor_betas_returned_for_all_strategies(strategy_name):
    """Verify that factor_betas are included in the response across all strategies."""
    payload = {
        "optimization_strategy": strategy_name,
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 25.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 75.0},
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "factor_betas" in data
    factor_betas = data["factor_betas"]
    assert factor_betas is not None

    for portfolio_key in ["current_portfolio", "optimized_portfolio"]:
        assert portfolio_key in factor_betas
        betas = factor_betas[portfolio_key]
        assert "value" in betas and isinstance(betas["value"], float)
        assert "momentum" in betas and isinstance(betas["momentum"], float)
        assert "size" in betas and isinstance(betas["size"], float)


def test_factor_betas_scenario_6_momentum_increase():
    """Verify that Scenario 6 (Maximize Momentum) strictly increases momentum factor beta."""
    payload = {
        "optimization_strategy": "Optimize Factor Exposure",
        "securities": [
            {"ticker": "IEFA", "security_name": "IEFA", "allocation": 20.0},
            {"ticker": "GLD", "security_name": "GLD", "allocation": 20.0},
            {"ticker": "AGG", "security_name": "AGG", "allocation": 20.0},
            {"ticker": "VEA", "security_name": "VEA", "allocation": 20.0},
            {"ticker": "SPY", "security_name": "SPY", "allocation": 20.0},
        ],
        "constraints": {
            "momentum": "maximize"
        }
    }
    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    factor_betas = response.json()["factor_betas"]
    curr_mom = factor_betas["current_portfolio"]["momentum"]
    opt_mom = factor_betas["optimized_portfolio"]["momentum"]

    assert opt_mom > curr_mom
    assert curr_mom == 0.13
    assert opt_mom == 0.19
