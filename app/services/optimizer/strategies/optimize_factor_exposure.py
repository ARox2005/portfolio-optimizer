"""Optimize Factor Exposure portfolio optimization strategy (Scenario 6 / Bonus)."""
from typing import List, Tuple
import numpy as np
from scipy.optimize import linprog
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.data_loader import calculate_fund_factor_betas, load_fund_dividend_yields


class OptimizeFactorExposureStrategy(BaseOptimizerStrategy):
    """Finds the portfolio allocation that optimizes exposure to equity risk factors."""

    strategy_id = "optimize_factor_exposure"
    strategy_name = "Optimize Factor Exposure"
    description = (
        "Maximize or minimize exposure to equity risk factors (Momentum, Value, Size) "
        "subject to security weight limits and portfolio constraints (e.g. minimum dividend yield)."
    )

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)
        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        # 1. Load Factor Betas for requested funds
        betas_df, normalized_tickers = calculate_fund_factor_betas(tickers)

        # 2. Determine target exposure direction for factors
        # Supports momentum, value, size (can be specified at constraints top-level or under constraints.factors)
        momentum_target = None
        value_target = None
        size_target = None

        if request.constraints:
            if request.constraints.factors:
                momentum_target = request.constraints.factors.momentum or request.constraints.momentum
                value_target = request.constraints.factors.value or request.constraints.value
                size_target = request.constraints.factors.size or request.constraints.size
            else:
                momentum_target = request.constraints.momentum
                value_target = request.constraints.value
                size_target = request.constraints.size

        # If none specified, default to maximizing momentum as per Scenario 6 specification
        if not any([momentum_target, value_target, size_target]):
            momentum_target = "maximize"

        factor_targets = {
            "momentum": momentum_target,
            "value": value_target,
            "size": size_target,
        }

        # 3. Build composite factor exposure objective vector
        # Maximizing exposure: weight * beta -> objective coefficient is +beta
        # Minimizing exposure: weight * beta -> objective coefficient is -beta
        # In linprog, we minimize c^T w, so to maximize exposure we set c = -composite_obj
        composite_obj = np.zeros(n, dtype=float)
        for factor, target in factor_targets.items():
            if target == "maximize":
                factor_betas = betas_df.loc[normalized_tickers, factor].values
                composite_obj += factor_betas
            elif target == "minimize":
                factor_betas = betas_df.loc[normalized_tickers, factor].values
                composite_obj -= factor_betas

        c = -composite_obj

        # 4. Setup security weight bounds
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

        # 5. Formulate linear constraints for linprog
        # Budget constraint: sum(w) = 1.0
        A_eq = [[1.0] * n]
        b_eq = [1.0]

        A_ub = None
        b_ub = None

        # Optional Minimum Dividend Yield Constraint
        div_yields = None
        min_div = None
        if request.constraints and request.constraints.min_dividend_yield is not None:
            min_div = request.constraints.min_dividend_yield / 100.0
            div_yields = load_fund_dividend_yields(tickers)

            # Pre-check dividend yield feasibility
            lp_div = linprog(
                -div_yields,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if lp_div.success:
                max_achievable = -lp_div.fun
                if max_achievable < min_div - 1e-6:
                    raise ValueError(
                        f"Infeasible constraint: required minimum dividend yield of {min_div * 100:.2f}% "
                        f"cannot be met with current bounds (maximum achievable is {max_achievable * 100:.2f}%)."
                    )
            else:
                raise ValueError("Infeasible constraints: given bounds and budget cannot be simultaneously satisfied.")

            # Inequality constraint: sum(w * div_yields) >= min_div <=> -sum(w * div_yields) <= -min_div
            A_ub = [-div_yields]
            b_ub = [-min_div]

        # 6. Solve Linear Program using HiGHS
        lp_result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not lp_result.success:
            raise ValueError(
                f"Optimize Factor Exposure optimization failed or constraints are infeasible: {lp_result.message}"
            )

        # 7. Post-validation: ensure dividend constraint is satisfied if present
        if min_div is not None and div_yields is not None:
            achieved_yield = float(np.dot(lp_result.x, div_yields))
            if achieved_yield < (min_div - 1e-4):
                raise ValueError(
                    f"Optimization could not satisfy minimum dividend yield constraint of {min_div * 100:.2f}%."
                )

        # 8. Clean up and round weights to exact 100.00%
        scaled_weights = (lp_result.x / np.sum(lp_result.x)) * 100.0
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
