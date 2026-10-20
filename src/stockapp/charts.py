import pandas as pd
import plotly.graph_objects as go


def split_label(ratio: float) -> str:
    """Human-readable split ratio: 4.0 -> "4:1", 0.1 (reverse split) -> "1:10"."""
    return f"{ratio:g}:1" if ratio >= 1 else f"1:{1 / ratio:g}"


def price_chart(close: pd.Series, actions: pd.DataFrame | None = None, dividend_currency: str = "") -> go.Figure:
    """
    Close price line. With `actions` (columns date, type, value), splits inside the plotted range are drawn as
    dashed vertical lines and dividends as small dots on the price, each with the ratio or amount on hover.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
    if actions is not None and not actions.empty and not close.empty:
        in_range = actions[(actions["date"] >= close.index[0]) & (actions["date"] <= close.index[-1])]
        splits = in_range[in_range["type"] == "split"]
        if not splits.empty:
            low, high = float(close.min()), float(close.max())
            xs: list[object] = []
            ys: list[float | None] = []
            texts: list[str | None] = []
            for d, ratio in zip(splits["date"], splits["value"], strict=True):
                label = f"Split {split_label(float(ratio))} on {d.date()}"
                xs += [d, d, None]
                ys += [low, high, None]
                texts += [label, label, None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Split",
                    line={"dash": "dash", "width": 1},
                    hovertext=texts,
                    hoverinfo="text",
                )
            )
        dividends = in_range[in_range["type"] == "dividend"]
        if not dividends.empty:
            on_price = close.reindex(pd.DatetimeIndex(dividends["date"]), method="ffill")
            unit = f" {dividend_currency}" if dividend_currency else ""
            fig.add_trace(
                go.Scatter(
                    x=dividends["date"],
                    y=on_price.to_numpy(),
                    mode="markers",
                    name="Dividend",
                    marker={"size": 5},
                    hovertext=[
                        f"Dividend {v:g}{unit} on {d.date()}"
                        for d, v in zip(dividends["date"], dividends["value"], strict=True)
                    ],
                    hoverinfo="text",
                )
            )
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


def kelly_over_windows_chart(
    kelly_by_window: dict[int, float],
    bands: dict[int, tuple[float, float]] | None = None,
    title: str = "Kelly Fraction Across Lookback Windows",
) -> go.Figure:
    """
    Bar per window. `bands` maps window -> (low, high), drawn as asymmetric error bars around each bar,
    e.g. a bootstrap 5-95% range; windows missing from `bands` get no error bar. A bar outside its own
    band (possible for a point estimate vs. bootstrap percentiles) gets a zero-length bar on that side.
    """
    windows = list(kelly_by_window)
    values = list(kelly_by_window.values())
    error_y = None
    if bands:
        error_y = go.bar.ErrorY(
            type="data",
            symmetric=False,
            array=[max(bands[w][1] - v, 0.0) if w in bands else 0.0 for w, v in zip(windows, values, strict=True)],
            arrayminus=[max(v - bands[w][0], 0.0) if w in bands else 0.0 for w, v in zip(windows, values, strict=True)],
        )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=windows, y=values, name="Kelly Fraction", error_y=error_y))
    fig.update_layout(title=title, xaxis_title="Window (trading days)", yaxis_title="Fraction")
    return fig
