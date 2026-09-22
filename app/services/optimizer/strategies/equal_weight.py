"""Equal Weights portfolio optimization strategy."""
from typing import List
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy


class EqualWeightsStrategy(BaseOptimizerStrategy):
    """Assigns equal allocation to all input securities (w = 100/N %)."""

    strategy_id = "equal_weights"
    strategy_name = "Equal weights"
    description = "Assign equal allocation to all securities (w = 100/N %). Simple baseline where sum of weights equals 100%."

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)

        if n == 0:
            raise ValueError("Securities list cannot be empty.")

        target_weight = 100.0 / n

        for sec in securities:
            if sec.min_weight is not None and target_weight < sec.min_weight:
                raise ValueError(
                    f"Infeasible constraint: Equal weight of {round(target_weight, 2)}% is less than "
                    f"min_weight of {sec.min_weight}% for ticker '{sec.ticker}'."
                )
            if sec.max_weight is not None and target_weight > sec.max_weight:
                raise ValueError(
                    f"Infeasible constraint: Equal weight of {round(target_weight, 2)}% exceeds "
                    f"max_weight of {sec.max_weight}% for ticker '{sec.ticker}'."
                )

        base_w = round(target_weight, 2)
        optimized_weights: List[float] = [base_w] * n
        diff = round(100.0 - sum(optimized_weights), 2)

        if diff != 0.0:
            step = 0.01 if diff > 0 else -0.01
            num_steps = int(round(abs(diff) / 0.01))
            for i in range(min(num_steps, n)):
                optimized_weights[i] = round(optimized_weights[i] + step, 2)

        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        changes = self.calculate_changes(
            tickers=tickers,
            names=names,
            current_weights=current_weights,
            optimized_weights=optimized_weights
        )

        return OptimizationResponse(
            optimization_strategy=self.strategy_name,
            allocation_changes=changes,
            metadata={
                "strategy_id": self.strategy_id,
                "asset_count": n,
                "formula": "w = 100 / N %",
                "total_optimized_weight": round(sum(optimized_weights), 2)
            }
        )
