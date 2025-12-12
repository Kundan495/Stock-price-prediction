# Stock Price Prediction Web App (Streamlit + numpy/pandas)

This repository contains a minimal Streamlit web app demonstrating a simple stock price prediction workflow:

- fetch historical market data with `yfinance`
- create windowed features using `numpy` / `pandas`
- train a simple `LinearRegression` model (from `scikit-learn`)
- backtest and show metrics and a next-day prediction

## Files
- `app.py` — Streamlit app UI and orchestration
- `data.py` — data fetching (yfinance)
- `model.py` — feature creation, training, prediction, metrics
- `utils.py` — plotting and small helpers
- `requirements.txt` — Python dependencies

## Setup (local)
1. Clone the repo (or copy files).
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\\Scripts\\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes & Next steps
- This is a demo. For production:
  - Use more robust features (volumes, technical indicators, macro features).
  - Use more powerful models (random forests, XGBoost, LSTM/transformer).
  - Add walk-forward validation and better risk-aware evaluation.
  - Add caching of downloaded data and model artifacts.
- The model here predicts next-day close using only recent close prices and a linear model — it's intended as an educational starting point.

## License
MIT — feel free to reuse and extend.
