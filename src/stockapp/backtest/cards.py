from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from stockapp.backtest.walkforward import WalkForwardResult

RunWalkForward = Callable[[str], WalkForwardResult]


@dataclass(frozen=True)
class SignalCards:
    table: pd.DataFrame
    """One row per signal, sorted by out-of-sample Sharpe (best first); losers are never dropped."""
    trials: int
    """Parameter combinations evaluated across every signal and both baselines."""


def signal_cards(signal_names: list[str], run: RunWalkForward, buy_hold: str, kelly_buy_hold: str) -> SignalCards:
    """
    Out-of-sample card per signal next to buy-and-hold and Kelly buy-and-hold on the same span. The verdict
    "beats buy-and-hold out of sample" is yes when the signal's out-of-sample Sharpe exceeds buy-and-hold's;
    max drawdown is shown alongside so a lower-risk loser on Sharpe is still visible.
    """
    bh, kbh = run(buy_hold), run(kelly_buy_hold)
    bh_m, kbh_m = bh.out_of_sample.metrics, kbh.out_of_sample.metrics
    rows, trials = [], bh.trials + kbh.trials
    for name in signal_names:
        if name == buy_hold:
            continue
        wf = run(name)
        trials += wf.trials
        m = wf.out_of_sample.metrics
        rows.append(
            {
                "Signal": name,
                "OOS Sharpe": m["Sharpe"],
                "OOS CAGR": m["CAGR"],
                "OOS max drawdown": m["Max drawdown"],
                "Exposure": m["Exposure"],
                "Annual turnover": m["Annual turnover"],
                "Buy & hold Sharpe": bh_m["Sharpe"],
                "Buy & hold max DD": bh_m["Max drawdown"],
                "Kelly B&H Sharpe": kbh_m["Sharpe"],
                "Beats buy-and-hold OOS": "yes" if m["Sharpe"] > bh_m["Sharpe"] else "no",
            }
        )
    table = pd.DataFrame(rows).sort_values("OOS Sharpe", ascending=False).set_index("Signal")
    return SignalCards(table, trials)
