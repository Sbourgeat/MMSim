import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings


@dataclass
class MarketParams:
    """Parameters for realistic market simulation"""

    initial_price: float = 150.0
    volatility: float = 0.02  # Daily volatility
    drift: float = 0.0001  # Daily drift
    market_hours: bool = True  # Simulate trading hours only
    base_volume: int = 50000000  # Base daily volume
    volume_std: float = 0.3  # Volume variability
    jump_intensity: float = 0.05  # Jump probability per day
    jump_size_mean: float = 0.0  # Mean jump size
    jump_size_std: float = 0.02  # Jump size variability


class MarketSimulator:
    def __init__(self, params: [MarketParams] = None):
        self.params = params or MarketParams()
        self.rng = np.random.default_rng(42)  # For reproducibility

    def generate_price_path(self, n_days: int, n_intraday: int = 390) -> np.ndarray:
        """
        Generate realistic price path using geometric Brownian motion with jumps

        Args:
            n_days: Number of trading days to simulate
            n_intraday: Number of intraday data points (default: 390 for 1-min)

        Returns:
            Price array of shape (n_days * n_intraday,)
        """
        total_steps = n_days * n_intraday
        dt = 1.0 / (252 * n_intraday)  # Trading days per year

        # Generate Brownian motion
        brownian_increments = self.rng.normal(0, np.sqrt(dt), total_steps)

        # Add jumps
        jumps = self._generate_jumps(total_steps, dt)

        # Combine components
        price_path = np.exp(
            (self.params.drift - 0.5 * self.params.volatility**2) * dt
            + self.params.volatility * brownian_increments
            + jumps
        )

        # Apply market hours constraint if specified
        if self.params.market_hours:
            price_path = self._apply_market_hours(price_path, n_days, n_intraday)

        # Calculate cumulative product to get price series
        prices = self.params.initial_price * np.cumprod(price_path)

        return prices

    def generate_volume_series(self, n_days: int, n_intraday: int = 390) -> np.ndarray:
        """
        Generate realistic volume series correlated with price movements

        Args:
            n_days: Number of trading days
            n_intraday: Number of intraday data points

        Returns:
            Volume array
        """
        total_steps = n_days * n_intraday

        # Base volume with daily patterns
        base_volume_per_step = self.params.base_volume / n_intraday

        # Add intraday volume pattern (U-shaped: high at open/close)
        intraday_pattern = self._create_intraday_pattern(n_intraday)

        # Add random fluctuations
        volume_noise = self.rng.normal(1.0, self.params.volume_std, total_steps)
        volume_noise = np.abs(volume_noise)  # Ensure positive volumes

        # Combine components - repeat intraday pattern for all days
        intraday_pattern_full = np.tile(intraday_pattern, n_days)
        volume_series = base_volume_per_step * intraday_pattern_full * volume_noise

        if self.params.market_hours:
            volume_series = self._apply_market_hours(volume_series, n_days, n_intraday)

        return volume_series.astype(int)

    def create_ohlcv_data(self, n_days: int, n_intraday: int = 390) -> pd.DataFrame:
        """
        Create complete OHLCV data suitable for TDA analysis

        Args:
            n_days: Number of trading days
            n_intraday: Intraday granularity

        Returns:
            DataFrame with OHLCV columns and datetime index
        """
        # Generate price path
        prices = self.generate_price_path(n_days, n_intraday)

        # Generate volume
        volumes = self.generate_volume_series(n_days, n_intraday)

        # Create OHLC from fine-grained prices
        if n_intraday > 1:
            # Resample to create daily OHLC from intraday data
            n_points = n_days * n_intraday
            daily_data = []

            for day in range(n_days):
                day_start = day * n_intraday
                day_end = (day + 1) * n_intraday if day < n_days - 1 else n_points

                day_prices = prices[day_start:day_end]
                day_volumes = volumes[day_start:day_end]

                daily_data.append(
                    {
                        "Open": day_prices[0],
                        "High": np.max(day_prices),
                        "Low": np.min(day_prices),
                        "Close": day_prices[-1],
                        "Volume": np.sum(day_volumes),
                    }
                )

            df = pd.DataFrame(daily_data)
        else:
            # Direct OHLC from daily prices
            df = pd.DataFrame(
                {
                    "Open": prices,
                    "High": prices,
                    "Low": prices,
                    "Close": prices,
                    "Volume": volumes,
                }
            )

        # Add datetime index
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(len(df))]
        df.index = dates

        return df

    def _generate_jumps(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate jump component for price dynamics"""
        # Poisson process for jump timing
        jump_times = self.rng.random(n_steps) < (self.params.jump_intensity * dt)

        # Jump sizes
        n_jumps = np.sum(jump_times)
        if n_jumps > 0:
            jump_sizes = self.rng.normal(
                self.params.jump_size_mean, self.params.jump_size_std, n_jumps
            )
            jumps = np.zeros(n_steps)
            jumps[jump_times] = jump_sizes
        else:
            jumps = np.zeros(n_steps)

        return jumps

    def _create_intraday_pattern(self, n_intraday: int) -> np.ndarray:
        """Create realistic U-shaped intraday volume pattern"""
        times = np.linspace(0, 1, n_intraday)

        # U-shaped pattern: high at open (0) and close (1), low in middle (0.5)
        pattern = 1.5 - np.exp(-10 * (times - 0.5) ** 2)

        # Add some noise
        pattern += np.random.normal(0, 0.05, n_intraday)
        pattern = np.maximum(pattern, 0.1)  # Ensure positive

        return pattern

    def _apply_market_hours(
        self, data: np.ndarray, n_days: int, n_intraday: int
    ) -> np.ndarray:
        """Apply market hours constraint by zeroing out non-trading periods"""
        if n_intraday <= 1:  # No effect for daily data
            return data

        # For simplicity, assume all intraday points are during market hours
        # In reality, you might want to zero out certain times or handle weekends
        return data

    def calibrate_to_historical(
        self, historical_data: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calibrate simulator parameters to match historical data characteristics

        Args:
            historical_data: DataFrame with OHLCV data

        Returns:
            Dictionary of calibrated parameters
        """
        # Calculate historical statistics
        returns = historical_data["Close"].pct_change().dropna()

        calibrated_params = {
            "initial_price": float(historical_data["Close"].iloc[-1]),
            "volatility": float(returns.std() * np.sqrt(252)),  # Annualized
            "drift": float(returns.mean() * 252),  # Annualized
            "base_volume": int(historical_data["Volume"].mean()),
            "volume_std": float(
                historical_data["Volume"].std() / historical_data["Volume"].mean()
            ),
        }

        return calibrated_params


# Utility functions for testing and examples
def create_sample_market_data(
    n_days: int = 252, use_aapl_params: bool = True
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
            volatility=0.25,  # AAPL's historical volatility
            drift=0.15,  # AAPL's historical drift
            base_volume=50000000,  # AAPL's average daily volume
            volume_std=0.3,
            jump_intensity=0.05,
        )
    else:
        params = MarketParams()

    simulator = MarketSimulator(params)
    return simulator.create_ohlcv_data(n_days)


def validate_market_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate generated market data for realism

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with validation metrics
    """
    validation_results = {}

    # Basic checks
    validation_results["has_ohlcv"] = all(
        col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]
    )
    validation_results["no_nan_values"] = not df.isnull().any().any()
    validation_results["positive_prices"] = (
        (df[["Open", "High", "Low", "Close"]] > 0).all().all().all()
    )
    validation_results["positive_volume"] = (df["Volume"] > 0).all()

    # Realism checks
    validation_results["high_low_consistent"] = (df["High"] >= df["Low"]).all()
    validation_results["close_in_range"] = (
        (df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])
    ).all()
    validation_results["open_in_range"] = (
        (df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])
    ).all()

    # Statistical checks
    returns = df["Close"].pct_change().dropna()
    validation_results["return_volatility"] = float(returns.std())
    validation_results["price_volume_correlation"] = float(
        abs(df["Close"].corr(df["Volume"]))
    )

    return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    simulator = MarketSimulator()
    data = simulator.create_ohlcv_data(
        n_days=30, n_intraday=78
    )  # 30 days, 5-minute bars

    # Validate the data
    validation = validate_market_data(data)
    print("Validation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    # Print summary statistics
    print(f"\nGenerated Data Summary:")
    print(f"  Shape: {data.shape}")
    print(f"  Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
    print(f"  Average volume: {data['Volume'].mean():,.0f}")
    print(f"  Daily volatility: {data['Close'].pct_change().std():.4f}")
