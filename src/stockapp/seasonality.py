import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_returns(prices: pd.Series) -> pd.DataFrame:
    """
    Calendar-month returns as a year x month table: last price of each month / last price of the previous month - 1.
    The first month has no prior month-end and is omitted; the current month is month-to-date.
    """
    month_end = prices.resample("ME").last().dropna()
    returns = month_end.pct_change().dropna()
    dates = pd.DatetimeIndex(returns.index)
    table = pd.DataFrame({"year": dates.year, "month": dates.month, "ret": returns.to_numpy()})
    pivot = table.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    pivot.columns = MONTH_NAMES
    return pivot


def average_by_month(monthly: pd.DataFrame) -> pd.DataFrame:
    """Mean monthly return per calendar month and the number of years behind each mean."""
    return pd.DataFrame({"mean": monthly.mean(), "count": monthly.count()})
