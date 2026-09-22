"""Portfolio optimizer API endpoints."""
from typing import List
from fastapi import APIRouter, HTTPException, status
from app.schemas.portfolio import (
    OptimizationRequest,
    OptimizationResponse,
    StrategyInfo,
)
from app.services.optimizer.registry import optimizer_registry

router = APIRouter()


UNPROCESSABLE_STATUS = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize Portfolio Allocation",
    description="Calculates optimized weights and allocation changes based on the selected strategy."
)
def optimize_portfolio(request: OptimizationRequest) -> OptimizationResponse:
    """Execute optimization for a given portfolio and strategy."""
    try:
        strategy = optimizer_registry.get_strategy(request.optimization_strategy)
        return strategy.optimize(request)
    except ValueError as err:
        raise HTTPException(
            status_code=UNPROCESSABLE_STATUS,
            detail=str(err),
        )


@router.get(
    "/strategies",
    response_model=List[StrategyInfo],
    summary="List Optimization Strategies",
    description="Lists all supported optimization strategies, both active and roadmap.",
)
def list_strategies() -> List[StrategyInfo]:
    """Return available optimization strategies."""
    return optimizer_registry.list_strategies()
