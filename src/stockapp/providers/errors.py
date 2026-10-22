"""Typed market data errors. Their messages are shown to end users as they are (stockapp_ui.context.user_message),
so they must never include credentials, file paths or raw upstream responses.
"""


class DataError(Exception):
    """Base class for market data errors."""


class TickerNotFound(DataError, LookupError):
    """The provider has no data for this symbol. Also a LookupError, matching fetch_price_history's contract."""


class ProviderUnavailable(DataError):
    """The provider could not be reached or failed (timeout, connection error, server error)."""


class RateLimited(DataError):
    """The provider refused the request because of rate limiting."""
