"""Pydantic schemas for portfolio optimization requests and responses."""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator
from app.core.config import ALLOCATION_SUM_TOLERANCE


class SecurityInput(BaseModel):
    """Input representation of a single security in the portfolio."""
    ticker: str = Field(..., min_length=1, max_length=20, description="Ticker symbol (e.g. SPY, IEFA)")
    security_name: str = Field(..., min_length=1, description="Security name (e.g. State Street SPDR S&P 500 ETF Trust)")
    allocation: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current portfolio allocation percentage (0 to 100)",
    )
    min_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Optional minimum weight constraint percentage (0 to 100)"
    )
    max_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Optional maximum weight constraint percentage (0 to 100)"
    )

    @model_validator(mode="after")
    def validate_min_max_bounds(self) -> "SecurityInput":
        if self.min_weight is not None and self.max_weight is not None:
            if self.min_weight > self.max_weight:
                raise ValueError(
                    f"min_weight ({self.min_weight}%) cannot exceed max_weight ({self.max_weight}%) "
                    f"for ticker '{self.ticker}'."
                )
        return self


class FactorConstraints(BaseModel):
    """Optional factor-level objectives (e.g. maximize, minimize)."""
    momentum: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Momentum factor"
    )
    value: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Value factor"
    )
    size: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Size factor"
    )


class PortfolioConstraints(BaseModel):
    """Optional portfolio-level constraints and factor objectives."""
    min_dividend_yield: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Minimum portfolio dividend yield percentage (e.g., 2.50 for 2.5%)",
    )
    momentum: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Momentum factor"
    )
    value: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Value factor"
    )
    size: Optional[Literal["maximize", "minimize"]] = Field(
        default=None,
        description="Target exposure direction for Size factor"
    )
    factors: Optional[FactorConstraints] = Field(
        default=None,
        description="Optional nested factor objectives"
    )



class OptimizationRequest(BaseModel):
    """Portfolio optimization request body."""
    optimization_strategy: str = Field(
        ...,
        description="Name of the optimization strategy (e.g. 'Equal weights' or 'equal_weights')"
    )
    securities: List[SecurityInput] = Field(
        ...,
        min_length=1,
        description="List of securities with current allocations and optional constraints"
    )
    constraints: Optional[PortfolioConstraints] = Field(
        default=None,
        description="Optional portfolio-level constraints"
    )


    @model_validator(mode="after")
    def validate_allocations_and_uniqueness(self) -> "OptimizationRequest":
        tickers = [sec.ticker.strip().upper() for sec in self.securities]
        if len(tickers) != len(set(tickers)):
            duplicates = {t for t in tickers if tickers.count(t) > 1}
            raise ValueError(f"Duplicate ticker(s) found in securities list: {', '.join(duplicates)}")

        total_allocation = sum(sec.allocation for sec in self.securities)
        if abs(total_allocation - 100.0) > ALLOCATION_SUM_TOLERANCE:
            raise ValueError(
                f"Sum of allocations across all securities must equal 100%. "
                f"Current sum: {round(total_allocation, 4)}%."
            )

        return self


class AllocationChange(BaseModel):
    """Optimized weight and change for a security."""
    ticker: str = Field(..., description="Security ticker symbol")
    security_name: str = Field(..., description="Full security name")
    current_weight: float = Field(..., description="Current weight percentage")
    optimized_weight: float = Field(..., description="Optimized weight percentage")
    change: float = Field(..., description="Percentage point delta (optimized_weight - current_weight)")


class FactorBetaValues(BaseModel):
    """Systematic equity risk factor exposures (betas)."""
    value: float = Field(..., description="Value factor beta")
    momentum: float = Field(..., description="Momentum factor beta")
    size: float = Field(..., description="Size factor beta")


class PortfolioFactorBetas(BaseModel):
    """Factor betas for current and optimized portfolios."""
    current_portfolio: FactorBetaValues = Field(..., description="Factor betas of the current (input) portfolio")
    optimized_portfolio: FactorBetaValues = Field(..., description="Factor betas of the optimized portfolio")


class OptimizationResponse(BaseModel):
    """Output returned by the portfolio optimizer."""
    optimization_strategy: str = Field(..., description="Strategy applied")
    allocation_changes: List[AllocationChange] = Field(..., description="Security-level weight changes")
    factor_betas: Optional[PortfolioFactorBetas] = Field(
        default=None,
        description="Factor betas for both current and optimized portfolios"
    )


class StrategyInfo(BaseModel):
    """Information about a supported optimization strategy."""
    id: str
    name: str
    description: str
    status: str
