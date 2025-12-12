import plotly.graph_objects as go
import pandas as pd

def plot_actual_vs_pred(preds_df: pd.DataFrame):
    """
    Returns a Plotly figure plotting actual vs predicted prices (for the backtest/test set).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds_df.index, y=preds_df["actual"], mode="lines+markers", name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=preds_df.index, y=preds_df["predicted"], mode="lines+markers", name="Predicted"
    ))
    fig.update_layout(title="Actual vs Predicted (backtest)", xaxis_title="Date", yaxis_title="Close price")
    return fig

def style_metrics(metrics: dict) -> str:
    """
    Return simple HTML to display metrics in Streamlit.
    """
    return (
        f"<b>RMSE:</b> {metrics['rmse']:.4f} &nbsp; | &nbsp;"
        f"<b>MAPE:</b> {metrics['mape']:.4%} &nbsp; | &nbsp;"
        f"<b>Train samples:</b> {metrics['train_samples']} &nbsp; | &nbsp;"
        f"<b>Test samples:</b> {metrics['test_samples']}"
    )
