import numpy as np
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


REGIME_COLORS = {
    "low": "rgba(46, 125, 50, 0.15)",
    "normal": "rgba(158, 158, 158, 0.10)",
    "high": "rgba(198, 40, 40, 0.18)",
}


def volatility_chart(rolling: pd.Series, ewma: pd.Series) -> go.Figure:
    """Annualized 21-day rolling volatility and EWMA volatility over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name="21-day rolling"))
    fig.add_trace(go.Scatter(x=ewma.index, y=ewma.values, mode="lines", name="EWMA (λ = 0.94)"))
    fig.update_layout(
        title="Annualized Volatility", xaxis_title="Date", yaxis_title="Volatility", yaxis_tickformat=".0%"
    )
    return fig


def underwater_chart(prices: pd.Series) -> go.Figure:
    """Drawdown from the running peak (P_t / max(P_0..P_t) - 1) over time."""
    drawdown = prices / prices.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", fill="tozeroy", name="Drawdown"))
    fig.update_layout(title="Drawdown From Peak", xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return fig


def qq_chart(qq: pd.DataFrame, title: str) -> go.Figure:
    """Q-Q scatter (columns theoretical, sample) with the 45-degree reference line."""
    lo = float(min(qq["theoretical"].min(), qq["sample"].min()))
    hi = float(max(qq["theoretical"].max(), qq["sample"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq["theoretical"], y=qq["sample"], mode="markers", marker={"size": 3}, name="Returns"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash"}, name="Perfect fit"))
    fig.update_layout(title=title, xaxis_title="Theoretical quantile", yaxis_title="Sample quantile")
    return fig


def acf_chart(acf_returns: np.ndarray, acf_squared: np.ndarray, band: float) -> go.Figure:
    """Autocorrelation of returns and of squared returns at lags 1..n with +/- band lines."""
    lags = list(range(1, len(acf_returns) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=acf_returns, name="Returns"))
    fig.add_trace(go.Bar(x=lags, y=acf_squared, name="Squared returns"))
    for level in (band, -band):
        fig.add_hline(y=level, line_dash="dash", line_width=1)
    fig.update_layout(title="Autocorrelation (lags 1–20)", xaxis_title="Lag (days)", yaxis_title="Autocorrelation")
    return fig


def regime_chart(prices: pd.Series, labels: pd.Series, title: str) -> go.Figure:
    """Price line with the background shaded by regime (one rectangle per contiguous run of the same label)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines", name="Price"))
    known = labels.dropna()
    shapes = []
    if not known.empty:
        run_id = (known != known.shift()).cumsum()
        for _, run in known.groupby(run_id):
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": run.index[0],
                    "x1": run.index[-1],
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": REGIME_COLORS.get(str(run.iloc[0]), "rgba(0, 0, 0, 0)"),
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
    for name, color in REGIME_COLORS.items():  # legend entries for the shading
        if name in set(known):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", marker={"color": color, "size": 12}, name=name)
            )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", shapes=shapes)
    return fig


def factor_chart(coefs: pd.Series, std_errors: pd.Series, labels: dict[str, str]) -> go.Figure:
    """Factor loadings as bars with +/- 2 standard-error bars."""
    names = [labels.get(str(n), str(n)) for n in coefs.index]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names, y=coefs.values, error_y=go.bar.ErrorY(type="data", array=(2 * std_errors).values), name="Loading"
        )
    )
    fig.update_layout(title="Factor Loadings (±2 SE)", xaxis_title="Factor", yaxis_title="Loading")
    return fig


def growth_chart(growth: pd.DataFrame) -> go.Figure:
    """Growth of $1 for each column."""
    fig = go.Figure()
    for column in growth.columns:
        fig.add_trace(go.Scatter(x=growth.index, y=growth[column].values, mode="lines", name=str(column)))
    fig.update_layout(title="Growth of $1", xaxis_title="Date", yaxis_title="Value of $1")
    return fig


def relative_chart(rolling: pd.DataFrame, benchmark: str) -> go.Figure:
    """Rolling relative return (stock minus benchmark) for each window."""
    fig = go.Figure()
    for column in rolling.columns:
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling[column].values, mode="lines", name=str(column)))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        title=f"Rolling Return Relative to {benchmark}",
        xaxis_title="Date",
        yaxis_title="Stock return − benchmark return",
        yaxis_tickformat=".0%",
    )
    return fig


def seasonality_heatmap(monthly: pd.DataFrame) -> go.Figure:
    """Year x month heatmap of monthly returns, centred on zero."""
    fig = go.Figure(
        go.Heatmap(
            z=monthly.values,
            x=list(monthly.columns),
            y=[str(y) for y in monthly.index],
            colorscale="RdYlGn",
            zmid=0,
            hovertemplate="%{y} %{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly Returns", xaxis_title="Month", yaxis_title="Year")
    return fig


def month_average_chart(average: pd.DataFrame) -> go.Figure:
    """Average return per calendar month, with the number of years in the hover text."""
    fig = go.Figure(
        go.Bar(
            x=list(average.index),
            y=average["mean"].values,
            hovertext=[f"{int(c)} years" for c in average["count"]],
            name="Average",
        )
    )
    fig.update_layout(
        title="Average Return by Calendar Month",
        xaxis_title="Month",
        yaxis_title="Average return",
        yaxis_tickformat=".1%",
    )
    return fig
