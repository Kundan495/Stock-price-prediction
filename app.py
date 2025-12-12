import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from data import fetch_data
from model import train_and_predict
from utils import plot_actual_vs_pred, style_metrics

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("Stock Price Prediction (Simple Windowed Linear Regression)")
st.markdown(
    "This demo fetches historical price data with yfinance, builds windowed features "
    "using numpy/pandas, trains a LinearRegression model, and shows backtest predictions "
    "and a next-day forecast."
)

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper()
    today = date.today()
    default_start = today - timedelta(days=365 * 2)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)
    window = st.number_input("Window size (days)", min_value=1, max_value=60, value=5, step=1)
    test_size_days = st.number_input("Backtest days (last N days as test)", min_value=5, max_value=180, value=60, step=1)
    retrain = st.button("Fetch & Train")

if retrain:
    with st.spinner(f"Downloading {ticker} data..."):
        df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error("No data returned. Check the ticker and date range.")
    else:
        st.subheader(f"{ticker} — Price data ({len(df)} rows)")
        st.line_chart(df["Close"])

        st.write("Head of data:")
        st.dataframe(df.tail(10))

        with st.spinner("Training model and predicting..."):
            try:
                results = train_and_predict(df["Close"], window=window, test_size_days=int(test_size_days))
            except ValueError as e:
                st.error(f"Unable to train model: {e}")
                results = None
            except Exception as e:
                st.error("An unexpected error occurred during training. See details:")
                st.exception(e)
                results = None

        if results is not None:
            preds_df = results["predictions_df"]
            metrics = results["metrics"]
            next_pred = results["next_day_prediction"]

            st.subheader("Backtest results")
            st.write("Metrics (on the held-out backtest set):")
            st.markdown(style_metrics(metrics), unsafe_allow_html=True)

            st.plotly_chart(plot_actual_vs_pred(preds_df), use_container_width=True)

            st.subheader("Next-day forecast")
            st.markdown(
                f"Model predicts next-day close (based on last {window} days): **{next_pred:.4f} {''}**"
            )

            st.info("This is a simple demo model. For production you'd use richer features, better models (e.g., tree ensembles, LSTM), and robust validation.")

else:
    st.info("Configure ticker, date range and parameters in the sidebar and click 'Fetch & Train' to begin.")
