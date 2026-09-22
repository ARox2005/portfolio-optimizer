from app.core.config import DATA_DIR
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

DATA_PATH = DATA_DIR / "Data.xlsx"

def load_fund_returns(tickers: List[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Returns")
    tickers_upper = [t.strip().upper() for t in tickers]
    df_filtered = df[df["ticker"].str.upper().isin(tickers_upper)]

    if df_filtered.empty:
        raise ValueError(f"None of the provided tickers {tickers} were found in the dataset.")
    
    pivot_df = df_filtered.pivot(index="date", columns="ticker", values="total_return").dropna()

    missing = set(tickers_upper) - set(pivot_df.columns)
    if missing:
        raise ValueError(f"Tickers not found in dataset: {', '.join(missing)}")

    pivot_df = pivot_df[tickers_upper]
    cov_matrix = pivot_df.cov().values

    return cov_matrix, tickers_upper

def load_fund_volatilities(tickers: List[str]) -> List[float]:
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Returns")
    tickers_upper = [t.strip().upper() for t in tickers]
    df_filtered = df[df["ticker"].str.upper().isin(tickers_upper)]

    if df_filtered.empty:
        raise ValueError(f"None of the provided tickers {tickers} were found in the dataset.")

    pivot_df = df_filtered.pivot(index="date", columns="ticker", values="total_return").dropna()

    missing = set(tickers_upper) - set(pivot_df.columns)
    if missing:
        raise ValueError(f"Tickers not found in dataset: {', '.join(missing)}")

    pivot_df = pivot_df[tickers_upper]
    # Standard deviation (volatility) of each ticker
    return [float(pivot_df[col].std()) for col in tickers_upper]


def load_fund_return_matrix(tickers: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Load aligned historical daily return matrix (T x N) for requested tickers."""
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Returns")
    tickers_upper = [t.strip().upper() for t in tickers]
    df_filtered = df[df["ticker"].str.upper().isin(tickers_upper)]

    if df_filtered.empty:
        raise ValueError(f"None of the provided tickers {tickers} were found in the dataset.")

    pivot_df = df_filtered.pivot(index="date", columns="ticker", values="total_return").dropna()

    missing = set(tickers_upper) - set(pivot_df.columns)
    if missing:
        raise ValueError(f"Tickers not found in dataset: {', '.join(missing)}")

    pivot_df = pivot_df[tickers_upper]
    return pivot_df.values, tickers_upper


def load_fund_dividend_yields(tickers: List[str]) -> np.ndarray:
    """Load dividend yields (as decimals) for requested tickers in order from Fund Info sheet."""
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Info")
    tickers_upper = [t.strip().upper() for t in tickers]
    df["ticker_upper"] = df["ticker"].astype(str).str.strip().str.upper()
    df_filtered = df[df["ticker_upper"].isin(tickers_upper)].set_index("ticker_upper")

    missing = set(tickers_upper) - set(df_filtered.index)
    if missing:
        raise ValueError(f"Tickers not found in Fund Info sheet: {', '.join(missing)}")

    yields = []
    for t in tickers_upper:
        val = df_filtered.loc[t, "dividend_yield"]
        yields.append(0.0 if pd.isna(val) else float(val))
    return np.array(yields, dtype=float)


def load_factor_returns() -> pd.DataFrame:
    """Load aligned historical daily factor returns (Momentum, Value, Size)."""
    df = pd.read_excel(DATA_PATH, sheet_name="Factor Returns")
    pivot_factors = df.pivot(index="date", columns="index_ticker", values="total_return").dropna()
    factor_cols = {
        "Momentum Factor": "momentum",
        "Value Factor": "value",
        "Size Factor": "size",
    }
    pivot_factors = pivot_factors.rename(columns=factor_cols)
    available_cols = [c for c in ["momentum", "value", "size"] if c in pivot_factors.columns]
    return pivot_factors[available_cols]


def calculate_fund_factor_betas(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate factor betas for requested tickers by regressing against historical factor returns.
    Returns:
        betas_df: DataFrame with tickers as index and columns ['alpha', 'momentum', 'value', 'size']
        tickers_upper: list of normalized ticker strings
    """
    tickers_upper = [t.strip().upper() for t in tickers]
    df_funds = pd.read_excel(DATA_PATH, sheet_name="Fund Returns")
    df_filtered = df_funds[df_funds["ticker"].str.upper().isin(tickers_upper)]

    if df_filtered.empty:
        raise ValueError(f"None of the provided tickers {tickers} were found in the dataset.")

    pivot_funds = df_filtered.pivot(index="date", columns="ticker", values="total_return").dropna()
    missing = set(tickers_upper) - set(pivot_funds.columns)
    if missing:
        raise ValueError(f"Tickers not found in Fund Returns: {', '.join(missing)}")

    pivot_funds = pivot_funds[tickers_upper]
    pivot_factors = load_factor_returns()

    common_dates = pivot_funds.index.intersection(pivot_factors.index)
    if len(common_dates) < 30:
        raise ValueError(f"Insufficient overlapping dates ({len(common_dates)}) between fund and factor returns.")

    funds_aligned = pivot_funds.loc[common_dates]
    factors_aligned = pivot_factors.loc[common_dates]

    factor_names = ["momentum", "value", "size"]
    X = factors_aligned[factor_names].values
    X_const = np.column_stack([np.ones(len(X)), X])

    betas_records = []
    for t in tickers_upper:
        y = funds_aligned[t].values
        coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
        betas_records.append({
            "ticker": t,
            "alpha": float(coeffs[0]),
            "momentum": float(coeffs[1]),
            "value": float(coeffs[2]),
            "size": float(coeffs[3]),
        })

    betas_df = pd.DataFrame(betas_records).set_index("ticker")
    return betas_df, tickers_upper


def calculate_portfolio_factor_betas(
    tickers: List[str],
    current_weights: List[float],
    optimized_weights: List[float],
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Calculate systematic risk factor betas (value, momentum, size) for both
    current and optimized portfolios via OLS regression over common historical dates.

    Args:
        tickers: List of security tickers.
        current_weights: Current portfolio allocations (percentages 0-100 or fractions).
        optimized_weights: Optimized portfolio allocations (percentages 0-100 or fractions).

    Returns:
        Dictionary containing 'current_portfolio' and 'optimized_portfolio' factor betas,
        or None if factor returns cannot be computed.
    """
    try:
        betas_df, normalized_tickers = calculate_fund_factor_betas(tickers)
    except Exception:
        return None

    w_curr = np.array(current_weights, dtype=float)
    if np.sum(w_curr) > 0:
        w_curr = w_curr / np.sum(w_curr)

    w_opt = np.array(optimized_weights, dtype=float)
    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)

    factors = ["value", "momentum", "size"]
    fund_betas = betas_df.loc[normalized_tickers, factors].values  # shape (N, 3)

    curr_betas = np.dot(w_curr, fund_betas)
    opt_betas = np.dot(w_opt, fund_betas)

    return {
        "current_portfolio": {
            "value": round(float(curr_betas[0]), 2),
            "momentum": round(float(curr_betas[1]), 2),
            "size": round(float(curr_betas[2]), 2),
        },
        "optimized_portfolio": {
            "value": round(float(opt_betas[0]), 2),
            "momentum": round(float(opt_betas[1]), 2),
            "size": round(float(opt_betas[2]), 2),
        },
    }

