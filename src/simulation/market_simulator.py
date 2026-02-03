import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional


@dataclass(frozen=True)
class MarketParams:
    """
    Parameters are DAILY unless stated otherwise.
    drift and volatility are daily (not annualized).
    jump_intensity is expected number of jumps per day.
    """

    initial_price: float = 150.0
    volatility: float = 0.02
    drift: float = 0.0001

    base_volume: int = 50_000_000
    volume_std: float = 0.3

    jump_intensity: float = 0.05
    jump_size_mean: float = 0.0
    jump_size_std: float = 0.02

    market_hours: bool = True  # kept for future, no effect in this simplified version


class MarketSimulator:
    def __init__(self, params: Optional[MarketParams] = None, seed: int = 42):
        self.params = params or MarketParams()
        self.rng = np.random.default_rng(seed)

    def generate_intraday_prices(
        self, n_days: int, n_intraday: int = 390
    ) -> np.ndarray:
        """
        Intraday GBM with jumps, consistent with DAILY drift and DAILY volatility.

        We discretize days into n_intraday steps:
          dt = 1 / n_intraday  day fraction
          log return step = (mu - 0.5 sigma^2) dt + sigma sqrt(dt) eps + jump
        """
        total_steps = n_days * n_intraday
        dt = 1.0 / n_intraday

        eps = self.rng.normal(0.0, 1.0, total_steps)
        brownian = self.params.volatility * np.sqrt(dt) * eps
        drift = (self.params.drift - 0.5 * self.params.volatility**2) * dt

        jumps = self._generate_jumps(total_steps=total_steps, n_intraday=n_intraday)

        log_increments = drift + brownian + jumps
        prices = self.params.initial_price * np.exp(np.cumsum(log_increments))
        return prices

    def generate_intraday_volumes(
        self, prices: np.ndarray, n_days: int, n_intraday: int = 390
    ) -> np.ndarray:
        """
        Intraday volume with U shape + noise + coupling to absolute intraday returns.
        """
        total_steps = n_days * n_intraday
        base_per_step = self.params.base_volume / n_intraday

        pattern = self._create_intraday_pattern(n_intraday)
        pattern_full = np.tile(pattern, n_days)

        noise = self.rng.normal(1.0, self.params.volume_std, total_steps)
        noise = np.clip(np.abs(noise), 0.05, None)

        # Couple to absolute intraday log returns
        logp = np.log(prices)
        abs_lr = np.abs(np.diff(logp, prepend=logp[0]))
        scale = 1.0 + 6.0 * (abs_lr / (abs_lr.mean() + 1e-12))

        vol = base_per_step * pattern_full * noise * scale
        vol = np.clip(vol, 1.0, None).astype(int)
        return vol

    def create_ohlcv_data(self, n_days: int, n_intraday: int = 390) -> pd.DataFrame:
        prices = self.generate_intraday_prices(n_days=n_days, n_intraday=n_intraday)
        volumes = self.generate_intraday_volumes(
            prices, n_days=n_days, n_intraday=n_intraday
        )

        daily = self._aggregate_to_daily_ohlcv(prices, volumes, n_days, n_intraday)

        start_date = datetime(2024, 1, 1)
        daily.index = [start_date + timedelta(days=i) for i in range(len(daily))]
        return daily

    def create_intraday_series(
        self, n_days: int, n_intraday: int = 390
    ) -> tuple[np.ndarray, np.ndarray]:
        prices = self.generate_intraday_prices(n_days=n_days, n_intraday=n_intraday)
        volumes = self.generate_intraday_volumes(
            prices, n_days=n_days, n_intraday=n_intraday
        )
        return prices, volumes

    def _aggregate_to_daily_ohlcv(
        self, prices: np.ndarray, volumes: np.ndarray, n_days: int, n_intraday: int
    ) -> pd.DataFrame:
        rows = []
        for d in range(n_days):
            s = d * n_intraday
            e = (d + 1) * n_intraday
            p = prices[s:e]
            v = volumes[s:e]
            rows.append(
                {
                    "Open": float(p[0]),
                    "High": float(np.max(p)),
                    "Low": float(np.min(p)),
                    "Close": float(p[-1]),
                    "Volume": int(np.sum(v)),
                }
            )
        return pd.DataFrame(rows)

    def _generate_jumps(self, total_steps: int, n_intraday: int) -> np.ndarray:
        """
        jump_intensity is expected number of jumps per day.
        Per step probability = jump_intensity / n_intraday.
        """
        p = self.params.jump_intensity / n_intraday
        jump_times = self.rng.random(total_steps) < p

        jumps = np.zeros(total_steps)
        n_jumps = int(jump_times.sum())
        if n_jumps > 0:
            jump_sizes = self.rng.normal(
                self.params.jump_size_mean, self.params.jump_size_std, n_jumps
            )
            jumps[jump_times] = jump_sizes
        return jumps

    def _create_intraday_pattern(self, n_intraday: int) -> np.ndarray:
        """
        U shaped volume pattern: high open and close, low mid day
        """
        t = np.linspace(0.0, 1.0, n_intraday)
        pattern = 1.5 - np.exp(-10.0 * (t - 0.5) ** 2)
        pattern += self.rng.normal(0.0, 0.05, n_intraday)
        pattern = np.maximum(pattern, 0.1)
        return pattern

    def calibrate_to_historical(
        self, historical_data: pd.DataFrame
    ) -> dict[str, float]:
        returns = historical_data["Close"].pct_change().dropna()

        return {
            "initial_price": float(historical_data["Close"].iloc[-1]),
            "volatility": float(returns.std()),  # daily
            "drift": float(returns.mean()),  # daily
            "base_volume": int(historical_data["Volume"].mean()),
            "volume_std": float(
                historical_data["Volume"].std() / historical_data["Volume"].mean()
            ),
        }


def validate_market_data(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["has_ohlcv"] = all(
        c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]
    )
    out["no_nan_values"] = not df.isnull().any().any()
    out["positive_prices"] = (df[["Open", "High", "Low", "Close"]] > 0).all().all()
    out["positive_volume"] = (df["Volume"] > 0).all()
    out["high_low_consistent"] = (df["High"] >= df["Low"]).all()
    out["close_in_range"] = (
        (df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])
    ).all()
    out["open_in_range"] = (
        (df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])
    ).all()
    returns = df["Close"].pct_change().dropna()
    out["return_volatility_daily"] = float(returns.std())
    out["price_volume_correlation_abs"] = float(abs(df["Close"].corr(df["Volume"])))
    return out


def create_sample_market_data(
    n_days: int = 50, use_aapl_params: bool = True
) -> pd.DataFrame:
    """
    Convenience function to create sample market data

    Args:
        n_days: Number of trading days
        use_aapl_params: Whether to use AAPL-like parameters

    Returns:
        DataFrame with OHLCV data
    """
    if use_aapl_params:
        # AAPL-like parameters
        params = MarketParams(
            initial_price=150.0,
            volatility=0.25,  # AAPL's historical volatility (daily)
            drift=0.15,  # AAPL's historical drift (daily)
            base_volume=50000000,  # AAPL's average daily volume
            volume_std=0.3,
            jump_intensity=0.05,
        )
    else:
        params = MarketParams()

    simulator = MarketSimulator(params)
    return simulator.create_ohlcv_data(n_days)
