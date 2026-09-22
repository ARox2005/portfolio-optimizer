"""Minimize Volatility portfolio optimization strategy."""
from typing import List
import numpy as np
from scipy.optimize import minimize
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.data_loader import load_fund_returns


class MinimizeVolatilityStrategy(BaseOptimizerStrategy):
    """Finds the portfolio allocation that produces the lowest volatility (standard deviation)."""

    strategy_id = "minimize_volatility"
    strategy_name = "Minimize Volatility"
    description = "Find the allocation that produces the lowest portfolio volatility (standard deviation of returns)."

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)
        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        # 1. Load covariance matrix for requested tickers
        cov_matrix, _ = load_fund_returns(tickers)

        # 2. Objective function: Minimize portfolio variance (w^T * Cov * w)
        # Annualized scaling factor (252) provides well-behaved numerical gradients
        scaled_cov = cov_matrix * 252.0

        def portfolio_variance_objective(w: np.ndarray) -> float:
            return float(w @ scaled_cov @ w)

        # 3. Setup bounds based on min_weight and max_weight
        bounds = []
        for sec in securities:
            lower = (sec.min_weight / 100.0) if sec.min_weight is not None else 0.0
            upper = (sec.max_weight / 100.0) if sec.max_weight is not None else 1.0
            bounds.append((lower, upper))

        # Constraint: sum of weights = 1.0 (100%)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Initial feasible guess
        w0 = np.array([sec.allocation / 100.0 for sec in securities])
        for i, (low, high) in enumerate(bounds):
            if w0[i] < low or w0[i] > high:
                w0 = np.array([1.0 / n] * n)
                break

        result = minimize(
            portfolio_variance_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if not result.success:
            # Retry with equal weights start
            w0_fallback = np.array([1.0 / n] * n)
            result = minimize(
                portfolio_variance_objective,
                w0_fallback,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if not result.success:
                raise ValueError(f"Minimize Volatility optimization failed: {result.message}")

        # 4. Scale weights to percentage and round
        scaled_weights = (result.x / np.sum(result.x)) * 100.0
        optimized_weights = [round(w, 2) for w in scaled_weights]

        # Reconcile rounding to guarantee exact 100.00% sum
        diff = round(100.0 - sum(optimized_weights), 2)
        if diff != 0.0:
            optimized_weights[-1] = round(optimized_weights[-1] + diff, 2)

        # 5. Format response
        changes = self.calculate_changes(
            tickers=tickers,
            names=names,
            current_weights=current_weights,
            optimized_weights=optimized_weights,
        )

        return OptimizationResponse(
            optimization_strategy=self.strategy_name,
            allocation_changes=changes,
        )
