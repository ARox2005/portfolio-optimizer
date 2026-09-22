from app.core.config import DATA_DIR
from typing import List, Tuple
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