import pytest
import pandas as pd
from src.data.import_data import import_price_historical


@pytest.fixture
def sample_symbol():
    return "AAPL"


@pytest.fixture
def multiple_symbols():
    return ["AAPL", "MSFT"]


def test_import_single_symbol(sample_symbol):
    """Test import of single symbol data"""
    df = import_price_historical(sample_symbol)

    # Test basic structure
    assert not df.empty, "DataFrame should not be empty"
    assert isinstance(df, pd.DataFrame), "Should return pandas DataFrame"
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    # Test expected columns
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' should be present"

    # Test data types
    assert df["Close"].dtype in ["float64", "int64"], "Close prices should be numeric"
    assert df["Volume"].dtype in ["float64", "int64"], "Volume should be numeric"

    # Test logical constraints
    assert (df["High"] >= df["Low"]).all(), "High should always be >= Low"
    assert (df["High"] >= df["Close"]).all(), "High should always be >= Close"
    assert (df["Low"] <= df["Close"]).all(), "Low should always be <= Close"


def test_import_multiple_symbols(multiple_symbols):
    """Test import of multiple symbols"""
    symbols_str = " ".join(multiple_symbols)
    df = import_price_historical(symbols_str)

    assert not df.empty, "DataFrame should not be empty"
    assert isinstance(df, pd.DataFrame), "Should return pandas DataFrame"
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"


def test_invalid_symbol_error():
    """Test error handling for invalid symbol"""
    with pytest.raises(ValueError, match="No data found"):
        import_price_historical("INVALID_SYMBOL_12345")


def test_date_range_parameters(sample_symbol):
    """Test date range filtering"""
    from datetime import datetime, timedelta

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now() - timedelta(days=1)

    df = import_price_historical(
        sample_symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    assert not df.empty, "Should have data for the specified range"
    assert df.index[0] >= pd.to_datetime(start_date), (
        "Data should start after start_date"
    )
    assert df.index[-1] <= pd.to_datetime(end_date), "Data should end before end_date"


def test_data_validation(sample_symbol):
    """Test data quality and validation"""
    df = import_price_historical(sample_symbol)

    # Test for NaN values in critical columns
    critical_columns = ["Open", "High", "Low", "Close"]
    for col in critical_columns:
        assert not df[col].isna().any(), f"{col} should not contain NaN values"

    # Test for negative values in price/volume
    for col in critical_columns:
        assert (df[col] >= 0).sum() == len(df), (
            f"{col} should not contain negative values"
        )

    assert (df["Volume"] >= 0).sum() == len(df), (
        "Volume should not contain negative values"
    )


def test_different_intervals(sample_symbol):
    """Test different time intervals"""
    for interval in ["1d", "1wk", "1mo"]:
        df = import_price_historical(sample_symbol, interval=interval)

        assert not df.empty, f"Should have data for {interval} interval"
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

        # For weekly data, dates should be roughly 7 days apart
        if interval == "1wk" and len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            avg_days = time_diffs.dt.days.mean()
            assert 5 <= avg_days <= 10, (
                f"Weekly data should be ~7 days apart, got {avg_days}"
            )
