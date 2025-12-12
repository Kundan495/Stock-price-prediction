import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using yfinance and return a DataFrame indexed by date.
    Returns None or empty DataFrame on failure.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Ensure Date index and sort
        df = df.sort_index()
        return df
    except Exception as e:
        # In production log error
        return pd.DataFrame()
