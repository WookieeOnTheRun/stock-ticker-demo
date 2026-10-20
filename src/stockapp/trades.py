from dataclasses import dataclass

import pandas as pd

TAX_NOTE = "Taxes are not estimated: that needs your tax lots and jurisdiction."


@dataclass(frozen=True)
class TradePlan:
    table: pd.DataFrame
    """One row per ticker plus Cash: current, target, difference, trade (% of portfolio) and, if known, shares."""
    turnover: float
    """0.5 * sum |target - current| over all positions including cash (fraction of the portfolio)."""
    cost_fraction: float
    """turnover * cost_bps / 1e4, as a fraction of the portfolio."""


def trade_plan(
    current: dict[str, float],
    target: dict[str, float],
    cost_bps: float = 0.0,
    latest_prices: dict[str, float] | None = None,
    portfolio_value: float | None = None,
) -> TradePlan:
    """
    Trades from current to target weights. Cash is 1 - sum of the stock weights on each side. If latest prices and
    a portfolio value are given, the trade is also shown in shares: (target - current) * value / price.
    """
    tickers = list(dict.fromkeys([*target, *current]))
    cur = pd.Series({t: current.get(t, 0.0) for t in tickers}, dtype="float64")
    tgt = pd.Series({t: target.get(t, 0.0) for t in tickers}, dtype="float64")
    cur["Cash"], tgt["Cash"] = 1.0 - cur.sum(), 1.0 - tgt.sum()
    diff = tgt - cur
    table = pd.DataFrame(
        {"Current weight": cur, "Target weight": tgt, "Difference": diff, "Trade (% of portfolio)": diff}
    )
    if latest_prices is not None and portfolio_value:
        table["Trade (shares)"] = [
            diff[t] * portfolio_value / latest_prices[t] if t in latest_prices else float("nan") for t in table.index
        ]
    turnover = 0.5 * float(diff.abs().sum())
    return TradePlan(table=table, turnover=turnover, cost_fraction=turnover * cost_bps / 1e4)
