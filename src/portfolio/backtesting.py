import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Union, Iterable, Optional


def annualized_sharpe(
    daily_returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio from daily returns.
    rf_annual is annualized risk-free rate (e.g., 0.02 for 2%).
    """
    if len(daily_returns) == 0:
        return np.nan
    rf_daily = rf_annual / periods_per_year
    excess = daily_returns - rf_daily
    mu_d = excess.mean()
    sd_d = excess.std()
    if sd_d == 0 or np.isnan(sd_d):
        return np.nan
    return float((mu_d / sd_d) * np.sqrt(periods_per_year))


def _coerce_weights(
    weights: Union[Dict[str, float], pd.Series, Iterable[float]],
    asset_names: Iterable[str]
) -> pd.Series:
    """
    Make sure weights are a pd.Series indexed by asset_names and sum to 1.
    Accepts dict/Series (with asset keys) or iterable of floats (aligned to asset_names).
    """
    assets = list(asset_names)
    if isinstance(weights, dict):
        w = pd.Series({a: float(weights.get(a, 0.0)) for a in assets})
    elif isinstance(weights, pd.Series):
        w = pd.Series([float(weights.get(a, 0.0)) for a in assets], index=assets)
    else:
        w = pd.Series([float(x) for x in weights], index=assets)
    s = w.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    w = w / s
    if (w < -1e-12).any():
        raise ValueError("This backtester supports long-only weights (>= 0).")
    return w


def backtest_constant_weights(
    returns_df: pd.DataFrame,
    weights: Union[Dict[str, float], pd.Series, Iterable[float]],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> Dict:
    """
    Backtest a constant-weights long-only portfolio on daily returns.

    returns_df: DataFrame of daily returns with columns matching asset names.
    weights: dict/Series/iterable specifying target weights (sum normalized to 1).
    """
    assets = list(returns_df.columns)
    w = _coerce_weights(weights, assets)

    daily_port_ret = (returns_df[assets] * w.values).sum(axis=1)
    cum = (1.0 + daily_port_ret).cumprod()

    total_return = float(cum.iloc[-1] - 1.0) if len(cum) else np.nan
    sharpe = annualized_sharpe(daily_port_ret, rf_annual, periods_per_year)

    return {
        "weights": w,
        "daily_returns": daily_port_ret,
        "cumulative": cum,
        "start": returns_df.index[0] if len(returns_df) else None,
        "end": returns_df.index[-1] if len(returns_df) else None,
        "total_return": total_return,
        "sharpe": sharpe,
    }


def backtest_monthly_rebalance(
    returns_df: pd.DataFrame,
    weights: Union[Dict[str, float], pd.Series, Iterable[float]],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> Dict:
    """
    Backtest with monthly rebalancing to target weights (on first business day of each month).
    Long-only, fully invested.
    """
    assets = list(returns_df.columns)
    w = _coerce_weights(weights, assets)

    # Group by month periods and apply constant weights within each month
    month_periods = returns_df.index.to_period("M")
    parts = []
    for _, idx in returns_df.groupby(month_periods).groups.items():
        block = returns_df.loc[returns_df.index[idx], assets]
        part = (block * w.values).sum(axis=1)
        parts.append(part)

    daily_port_ret = pd.concat(parts).sort_index()
    cum = (1.0 + daily_port_ret).cumprod()

    total_return = float(cum.iloc[-1] - 1.0) if len(cum) else np.nan
    sharpe = annualized_sharpe(daily_port_ret, rf_annual, periods_per_year)

    return {
        "weights": w,
        "daily_returns": daily_port_ret,
        "cumulative": cum,
        "start": returns_df.index[0] if len(returns_df) else None,
        "end": returns_df.index[-1] if len(returns_df) else None,
        "total_return": total_return,
        "sharpe": sharpe,
    }


def build_benchmark_60_40(returns_df: pd.DataFrame) -> pd.Series:
    """
    Build a static 60/40 SPY/BND daily returns benchmark (0% TSLA).
    Expects columns to include 'SPY' and 'BND'.
    """
    missing = [c for c in ["SPY", "BND"] if c not in returns_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for benchmark: {missing}")
    w_bench = pd.Series({"SPY": 0.60, "BND": 0.40})
    bench_ret = (returns_df[w_bench.index] * w_bench.values).sum(axis=1)
    return bench_ret


def plot_cumulative(
    cum_strategy: pd.Series,
    cum_benchmark: pd.Series,
    title: str = "Strategy vs Benchmark — Cumulative Return",
    out_path: Optional[str] = None,
):
    """
    Plot and optionally save cumulative return curves for strategy and benchmark.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy.index, cum_strategy.values, label="Strategy", color="tab:blue", linewidth=2)
    plt.plot(cum_benchmark.index, cum_benchmark.values, label="Benchmark (60/40)", color="tab:orange", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (Start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()