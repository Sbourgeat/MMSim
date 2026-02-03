import pandas as pd
import yfinance as yf


def import_price_historical(
    symbol: str, start_date="2024-01-01", end_date=None, interval="1d"
) -> pd.DataFrame:
    """Import price data for a given symbol using yfinance

    Args:
        symbol: Stock ticker symbol(s)
        start_date: Start date for data (YYYY-MM-DD format or datetime)
        end_date: End date for data (YYYY-MM-DD format or datetime)
        interval: Data frequency ('1d', '1w', '1m' etc.)

    Returns:
        DataFrame with OHLCV data
    """
    df = yf.download(
        symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data found for symbol {symbol}")

    # Handle MultiIndex columns (common with single symbols)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # Keep only the price column names

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    return df
