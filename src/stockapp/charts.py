import pandas as pd
import plotly.graph_objects as go


def price_chart(close: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
    fig.update_layout(title="Close Price", xaxis_title="Date", yaxis_title="Price")
    return fig


def returns_histogram(returns_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns_df["simple_return"], nbinsx=50, name="Simple Returns"))
    fig.update_layout(title="Daily Return Distribution", xaxis_title="Return", yaxis_title="Count")
    return fig


def cumulative_return_chart(returns_df: pd.DataFrame) -> go.Figure:
    cum = (1 + returns_df["simple_return"]).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative Return"))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1")
    return fig


def kelly_over_windows_chart(kelly_by_window: dict[int, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(kelly_by_window), y=list(kelly_by_window.values()), name="Kelly Fraction"))
    fig.update_layout(
        title="Kelly Fraction Across Lookback Windows", xaxis_title="Window (trading days)", yaxis_title="Fraction"
    )
    return fig
