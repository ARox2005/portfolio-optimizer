"""Minimize Drawdown portfolio optimization strategy."""
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize, linprog
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.data_loader import load_fund_return_matrix, load_fund_dividend_yields


class MinimizeDrawdownStrategy(BaseOptimizerStrategy):
    """Finds the portfolio allocation that minimizes the maximum historical drawdown."""

    strategy_id = "minimize_drawdown"
    strategy_name = "Minimize Drawdown"
    description = "Find the portfolio allocation that minimizes the maximum historical drawdown over the given time period."

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)
        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        # 1. Load aligned daily returns matrix (T x N)
        returns_matrix, _ = load_fund_return_matrix(tickers)

        # 2. Objective function: Maximum Historical Drawdown
        def max_drawdown_objective(w: np.ndarray) -> float:
            port_returns = returns_matrix @ w
            wealth_index = np.cumprod(1.0 + port_returns)
            running_peak = np.maximum.accumulate(wealth_index)
            drawdowns = (running_peak - wealth_index) / running_peak
            return float(np.max(drawdowns))

        # 3. Setup bounds based on security-level min_weight and max_weight
        bounds: List[Tuple[float, float]] = []
        for sec in securities:
            lower = (sec.min_weight / 100.0) if sec.min_weight is not None else 0.0
            upper = (sec.max_weight / 100.0) if sec.max_weight is not None else 1.0
            bounds.append((lower, upper))

        # Check bounds sanity
        sum_lower = sum(b[0] for b in bounds)
        sum_upper = sum(b[1] for b in bounds)
        if sum_lower > 1.0 + 1e-6:
            raise ValueError(
                f"Infeasible constraints: sum of minimum weights ({sum_lower * 100:.2f}%) exceeds 100%."
            )
        if sum_upper < 1.0 - 1e-6:
            raise ValueError(
                f"Infeasible constraints: sum of maximum weights ({sum_upper * 100:.2f}%) is less than 100%."
            )

        # Constraint: sum of weights = 1.0 (100%)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Optional Minimum Dividend Yield Constraint
        div_yields = None
        min_div = None
        lp_res = None
        if request.constraints and request.constraints.min_dividend_yield is not None:
            min_div = request.constraints.min_dividend_yield / 100.0
            div_yields = load_fund_dividend_yields(tickers)

            lp_res = linprog(
                -div_yields,
                A_eq=[[1.0] * n],
                b_eq=[1.0],
                bounds=bounds,
                method="highs",
            )
            if lp_res.success:
                max_achievable = -lp_res.fun
                if max_achievable < min_div - 1e-6:
                    raise ValueError(
                        f"Infeasible constraint: required minimum dividend yield of {min_div * 100:.2f}% "
                        f"cannot be met with current bounds (maximum achievable is {max_achievable * 100:.2f}%)."
                    )
            else:
                raise ValueError("Infeasible constraints: given bounds and budget cannot be simultaneously satisfied.")

            constraints.append(
                {"type": "ineq", "fun": lambda w, dy=div_yields, md=min_div: float(np.dot(w, dy) - md)}
            )

        # Initial feasible guess
        w0 = np.array([sec.allocation / 100.0 for sec in securities])
        for i, (low, high) in enumerate(bounds):
            if w0[i] < low or w0[i] > high:
                w0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
                w0 = w0 / np.sum(w0)
                break

        if min_div is not None and div_yields is not None and lp_res is not None and lp_res.success:
            if np.dot(w0, div_yields) < min_div:
                w0 = lp_res.x

        result = minimize(
            max_drawdown_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-7, "maxiter": 500},
        )

        if not result.success:
            # Fallback retry with bounds center
            w0_fallback = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            w0_fallback = w0_fallback / np.sum(w0_fallback)
            result = minimize(
                max_drawdown_objective,
                w0_fallback,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-7, "maxiter": 500},
            )
            if not result.success:
                raise ValueError(f"Minimize Drawdown optimization failed: {result.message}")

        # Post-validation
        if min_div is not None and div_yields is not None:
            achieved_yield = float(np.dot(result.x, div_yields))
            if achieved_yield < (min_div - 1e-4):
                raise ValueError(
                    f"Optimization could not satisfy minimum dividend yield constraint of {min_div * 100:.2f}%."
                )

        # 4. Scale weights to percentage and round
        scaled_weights = (result.x / np.sum(result.x)) * 100.0
        optimized_weights = [round(w, 2) for w in scaled_weights]

        # Reconcile rounding to guarantee exact 100.00% sum
        diff = round(100.0 - sum(optimized_weights), 2)
        if diff != 0.0:
            max_idx = int(np.argmax(optimized_weights))
            optimized_weights[max_idx] = round(optimized_weights[max_idx] + diff, 2)

        # 5. Format response
        changes = self.calculate_changes(
            tickers=tickers,
            names=names,
            current_weights=current_weights,
            optimized_weights=optimized_weights,
        )

        factor_betas = self.calculate_factor_betas(
            tickers=tickers,
            current_weights=current_weights,
            optimized_weights=optimized_weights,
        )

        return OptimizationResponse(
            optimization_strategy=self.strategy_name,
            allocation_changes=changes,
            factor_betas=factor_betas,
        )
