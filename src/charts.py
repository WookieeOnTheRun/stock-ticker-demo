import pandas as pd
import plotly.graph_objs as go


def price_chart(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"], mode="lines", name="Close"))
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


def kelly_over_windows_chart(returns_df: pd.DataFrame, window_sizes: list[int], kelly_func) -> go.Figure:
    points = []
    for w in window_sizes:
        sub = returns_df.tail(w)
        kr = kelly_func(sub["simple_return"]) if not sub.empty else None
        points.append({"window": w, "kelly": kr.applied_kelly if kr else 0.0})
    df = pd.DataFrame(points)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["window"], y=df["kelly"], name="Kelly Fraction"))
    fig.update_layout(title="Kelly Fraction Across Lookback Windows", xaxis_title="Window (days)", yaxis_title="Fraction")
    return fig
