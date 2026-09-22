"""Maximize Sharpe Ratio portfolio optimization strategy."""
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize, linprog
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.data_loader import load_fund_return_matrix, load_fund_dividend_yields

# Historical 3-month US Treasury Bill rate benchmark (~2.0% balanced empirical calibration)
DEFAULT_RISK_FREE_RATE = 0.02


class MaximizeSharpeRatioStrategy(BaseOptimizerStrategy):
    """Finds the portfolio allocation that maximizes the Sharpe Ratio (risk-adjusted return)."""

    strategy_id = "maximize_sharpe_ratio"
    strategy_name = "Maximize Sharpe Ratio"
    description = (
        "Maximize portfolio risk-adjusted return (Sharpe Ratio) subject to optional "
        "security weight bounds and portfolio constraints (e.g. minimum dividend yield)."
    )

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)
        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        # 1. Load historical return matrix (T x N) and annualize statistics
        ret_matrix, _ = load_fund_return_matrix(tickers)
        mu = np.mean(ret_matrix, axis=0) * 252.0
        cov = np.cov(ret_matrix, rowvar=False) * 252.0

        rf = DEFAULT_RISK_FREE_RATE

        # 2. Setup security weight bounds
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

        # 3. Formulate constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Optional Minimum Dividend Yield Constraint
        div_yields = None
        min_div = None
        if request.constraints and request.constraints.min_dividend_yield is not None:
            min_div = request.constraints.min_dividend_yield / 100.0
            div_yields = load_fund_dividend_yields(tickers)

            # Pre-check feasibility using linear programming
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
            elif not lp_res.success:
                raise ValueError("Infeasible constraints: given bounds and budget cannot be simultaneously satisfied.")

            constraints.append(
                {"type": "ineq", "fun": lambda w, dy=div_yields, md=min_div: float(np.dot(w, dy) - md)}
            )

        # 4. Objective: Minimize Negative Sharpe Ratio
        def neg_sharpe_objective(w: np.ndarray) -> float:
            port_ret = float(np.dot(w, mu))
            port_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
            if port_vol <= 1e-9:
                return 0.0
            return -(port_ret - rf) / port_vol

        # 5. Initial guess
        w0 = np.array([sec.allocation / 100.0 for sec in securities])
        for i, (low, high) in enumerate(bounds):
            if w0[i] < low or w0[i] > high:
                w0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
                w0 = w0 / np.sum(w0)
                break

        # If dividend constraint is active and w0 doesn't satisfy it, use the LP feasible point
        if min_div is not None and div_yields is not None and lp_res.success:
            if np.dot(w0, div_yields) < min_div:
                w0 = lp_res.x

        # 6. Optimize using SLSQP
        result = minimize(
            neg_sharpe_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            # Fallback retry from center of bounds
            w0_fallback = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
            w0_fallback = w0_fallback / np.sum(w0_fallback)
            result = minimize(
                neg_sharpe_objective,
                w0_fallback,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

            if not result.success:
                raise ValueError(
                    f"Maximize Sharpe Ratio optimization failed or constraints are infeasible: {result.message}"
                )

        # 7. Post-validation: ensure dividend constraint is satisfied if present
        if min_div is not None and div_yields is not None:
            achieved_yield = float(np.dot(result.x, div_yields))
            if achieved_yield < (min_div - 1e-4):
                raise ValueError(
                    f"Optimization could not satisfy minimum dividend yield constraint of {min_div * 100:.2f}%."
                )

        # 8. Clean up and round weights
        scaled_weights = (result.x / np.sum(result.x)) * 100.0
        # Zero out negligible numerical residual allocations
        scaled_weights = np.where(scaled_weights < 0.005, 0.0, scaled_weights)
        scaled_weights = (scaled_weights / np.sum(scaled_weights)) * 100.0

        optimized_weights = [round(float(w), 2) for w in scaled_weights]

        # Reconcile rounding differences to enforce exact 100.00% sum
        diff = round(100.0 - sum(optimized_weights), 2)
        if diff != 0.0:
            max_idx = int(np.argmax(optimized_weights))
            optimized_weights[max_idx] = round(optimized_weights[max_idx] + diff, 2)

        # 9. Format and return response
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
