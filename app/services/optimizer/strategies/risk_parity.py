"""Risk Parity portfolio optimization strategy."""
from typing import List
import numpy as np
from scipy.optimize import minimize
from app.schemas.portfolio import OptimizationRequest, OptimizationResponse
from app.services.optimizer.base import BaseOptimizerStrategy
from app.services.data_loader import load_fund_returns, load_fund_volatilities


class RiskParityStrategy(BaseOptimizerStrategy):
    """Allocates weights such that each security contributes equally to total portfolio risk."""

    strategy_id = "risk_parity"
    strategy_name = "Risk Parity"
    description = "Allocate weights such that each security contributes equally to total portfolio risk."

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        securities = request.securities
        n = len(securities)
        tickers = [sec.ticker for sec in securities]
        names = [sec.security_name for sec in securities]
        current_weights = [sec.allocation for sec in securities]

        volatilities = load_fund_volatilities(tickers)

        inv_vols = [1.0 / v for v in volatilities]
        total_inv_vol = sum(inv_vols)
        raw_weights = [(iv / total_inv_vol) * 100.0 for iv in inv_vols]
        optimized_weights = [round(w, 2) for w in raw_weights]

        diff = round(100.0 - sum(optimized_weights), 2)
        if diff != 0.0:
            optimized_weights[-1] = round(optimized_weights[-1] + diff, 2)

        for sec, opt_w in zip(securities, optimized_weights):
            if sec.min_weight is not None and opt_w < sec.min_weight:
                raise ValueError(f"Optimized weight ({opt_w}%) violates min_weight ({sec.min_weight}%) for {sec.ticker}")
            if sec.max_weight is not None and opt_w > sec.max_weight:
                raise ValueError(f"Optimized weight ({opt_w}%) violates max_weight ({sec.max_weight}%) for {sec.ticker}")

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
