"""Registry for portfolio optimization strategies."""
from typing import Dict, List, Optional
from app.schemas.portfolio import StrategyInfo
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.optimizer.strategies.equal_weight import EqualWeightsStrategy
from app.services.optimizer.strategies.risk_parity import RiskParityStrategy


class StrategyRegistry:
    """Central registry and factory for optimization strategies."""

    def __init__(self) -> None:
        self._strategies: Dict[str, BaseOptimizerStrategy] = {}
        self._aliases: Dict[str, str] = {}
        self._roadmap: Dict[str, StrategyInfo] = {}
        self._register_builtins()

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Normalize strategy string to lowercase alphanumeric with underscores."""
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    def register(self, strategy: BaseOptimizerStrategy, aliases: Optional[List[str]] = None) -> None:
        """Register an active optimization strategy instance."""
        canonical_key = self._normalize_key(strategy.strategy_id)
        self._strategies[canonical_key] = strategy

        self._aliases[self._normalize_key(strategy.strategy_name)] = canonical_key

        if aliases:
            for alias in aliases:
                self._aliases[self._normalize_key(alias)] = canonical_key

    def _register_builtins(self) -> None:
        """Register initial and roadmap strategies."""
        eq_strategy = EqualWeightsStrategy()
        self.register(eq_strategy, aliases=["equal_weights", "equal weights", "equal-weights", "equal_weight", "equalweight"])
        rp_strategy = RiskParityStrategy()
        self.register(rp_strategy, aliases=["risk_parity", "risk parity", "Risk Parity"])

    def get_strategy(self, strategy_name: str) -> BaseOptimizerStrategy:
        """Retrieve strategy instance by name or alias."""
        key = self._normalize_key(strategy_name)
        canonical = self._aliases.get(key, key)

        if canonical in self._strategies:
            return self._strategies[canonical]

        supported = [s.strategy_name for s in self._strategies.values()]
        raise ValueError(
            f"Unsupported optimization strategy: '{strategy_name}'. "
            f"Available strategies: {', '.join(supported)}."
        )

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                id=s.strategy_id,
                name=s.strategy_name,
                description=s.description,
                status="active",
            )
            for s in self._strategies.values()
        ]



optimizer_registry = StrategyRegistry()
