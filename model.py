import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Provide a robust MAPE function that works even if older scikit-learn doesn't export it
try:
    from sklearn.metrics import mean_absolute_percentage_error as _mape
    def mean_absolute_percentage_error(y_true, y_pred):
        return float(_mape(y_true, y_pred))
except Exception:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        eps = 1e-8
        denom = np.maximum(np.abs(y_true), eps)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))


def create_window_features(series: pd.Series, window: int):
    """
    Turn a 1D series (prices) into windowed features for supervised learning.
    Each row uses the previous `window` closes to predict the next day's close.
    Returns X (2D numpy) and y (1D numpy) and the corresponding dates for y.
    """
    prices = series.dropna().values
    dates = series.dropna().index
    n = len(prices)
    if n <= window:
        return np.array([]), np.array([]), []

    X = []
    y = []
    y_dates = []
    for i in range(window, n):
        X.append(prices[i - window : i])
        y.append(prices[i])
        y_dates.append(dates[i])
    X = np.array(X)
    y = np.array(y)
    return X, y, y_dates


def train_and_predict(close_series: pd.Series, window: int = 5, test_size_days: int = 60):
    """
    Train a simple LinearRegression on windowed close prices.
    Splits data by time: last `test_size_days` rows are used as the test set.
    Returns predictions DataFrame, metrics dict, and a next-day prediction scalar.
    """
    X, y, y_dates = create_window_features(close_series, window)
    if X.size == 0:
        raise ValueError("Not enough data for the chosen window size.")

    # Build a DataFrame to align dates
    df_features = pd.DataFrame(X, index=y_dates)
    df_targets = pd.Series(y, index=y_dates, name="Target")

    # time-based split
    split_index = max(1, len(df_targets) - test_size_days)
    X_train = df_features.iloc[:split_index].values
    y_train = df_targets.iloc[:split_index].values
    X_test = df_features.iloc[split_index:].values
    y_test = df_targets.iloc[split_index:].values
    test_dates = df_targets.index[split_index:]

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "mape": mape,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }

    preds_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test,
        "predicted": y_pred
    }).set_index("date")

    # last window to forecast next day
    last_window = close_series.dropna().values[-window:]
    last_window_scaled = scaler.transform(last_window.reshape(1, -1))
    next_day_pred = float(model.predict(last_window_scaled)[0])

    return {"predictions_df": preds_df, "metrics": metrics, "next_day_prediction": next_day_pred}
