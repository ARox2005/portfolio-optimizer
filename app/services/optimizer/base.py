"""Base abstraction for portfolio optimization strategies."""
from abc import ABC, abstractmethod
from typing import List
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse, AllocationChange


class BaseOptimizerStrategy(ABC):
    """Abstract base class for all optimization strategies."""

    strategy_id: str
    strategy_name: str
    description: str

    @abstractmethod
    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        """Execute portfolio optimization strategy.

        Args:
            request: OptimizationRequest containing securities and constraints.

        Returns:
            OptimizationResponse containing calculated allocation changes.
        """
        pass

    @staticmethod
    def calculate_changes(
        tickers: List[str],
        names: List[str],
        current_weights: List[float],
        optimized_weights: List[float]
    ) -> List[AllocationChange]:
        """Compute allocation changes and percentage changes.

        Args:
            tickers: List of security tickers.
            names: List of security names.
            current_weights: List of original weights in %.
            optimized_weights: List of optimized weights in %.

        Returns:
            List of AllocationChange schema objects.
        """
        changes: List[AllocationChange] = []
        for ticker, name, curr_w, opt_w in zip(tickers, names, current_weights, optimized_weights):
            curr_w_rounded = round(curr_w, 2)
            opt_w_rounded = round(opt_w, 2)
            delta = round(opt_w_rounded - curr_w_rounded, 2)

            if curr_w_rounded > 0:
                rel_change = round((delta / curr_w_rounded) * 100.0, 2)
            else:
                rel_change = 0.0 if opt_w_rounded == 0.0 else 100.0

            changes.append(
                AllocationChange(
                    ticker=ticker,
                    security_name=name,
                    current_weight=curr_w_rounded,
                    optimized_weight=opt_w_rounded,
                    change=delta,
                )
            )
        return changes
