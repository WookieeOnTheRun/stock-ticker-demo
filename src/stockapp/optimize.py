"""Portfolio construction: unconstrained and constrained fractional Kelly (cvxpy), HRP and the frontier."""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

# Imported from their defining modules: cvxpy re-exports atoms with a star import that mypy cannot follow.
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.sum_squares import sum_squares
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

SOLVER = cp.CLARABEL  # interior point: meets constraints to ~1e-9; OSQP's default tolerance (~1e-4) is too loose
FRONTIER_POINTS = 25
ACCEPTED_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)  # inaccurate solutions must still pass the feasibility check
FEASIBILITY_TOL = 1e-6  # largest constraint violation, in weight units, for a solution to be used
RETRY_JITTER = 1e-6  # the retry adds this times Sigma's mean diagonal to its diagonal


def unconstrained_kelly(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Full multi-asset Kelly w* = Sigma^-1 (mu - r), computed with numpy.linalg.solve (never an explicit inverse).
    mu is the excess return. Unconstrained: may include shorts and leverage; shown for reference only.
    """
    return np.linalg.solve(sigma, mu)


@dataclass(frozen=True)
class Allocation:
    """Constrained Kelly weights with the cash remainder and solver status, or the reason there are none."""

    weights: np.ndarray | None
    """None if the optimizer failed: there is then no recommendation, never a substitute portfolio."""
    cash: float | None
    """1 - sum(weights): held in the risk-free asset; None if the optimizer failed."""
    status: str
    """cvxpy status of the solution used ("optimal" or "optimal_inaccurate"), or "failed"."""
    reason: str | None
    """Why the optimizer failed; None if it succeeded."""

    @property
    def failed(self) -> bool:
        """True if there are no weights because the optimizer failed."""
        return self.weights is None


class ConstrainedKelly:
    """
    Fractional Kelly with constraints, as a QP compiled once and re-solved with new mu / Sigma / c:
      maximize  w'mu - 1/(2c) w'Sigma w
      s.t.      w >= 0 (long only), sum(w) <= gross_max (sum |w| if not long only), w_i <= w_max,
                sum of w over each sector <= its cap.
    The quadratic term is written as ||sqrt(1/(2c)) L' w||^2 with Sigma = L L' (Cholesky), which keeps the problem
    DPP so re-solves skip recompilation. The remainder 1 - sum(w) is cash at the risk-free rate.
    """

    def __init__(
        self,
        n: int,
        long_only: bool = True,
        gross_max: float = 1.0,
        w_max: float = 0.25,
        sector_caps: dict[str, float] | None = None,
        sectors: list[str | None] | None = None,
    ) -> None:
        """Build the parametrized problem for n assets; sectors[i] is asset i's sector name (None: no sector cap)."""
        self.n, self.gross_max = n, gross_max
        self._w = cp.Variable(n)
        self._mu = cp.Parameter(n)
        self._risk = cp.Parameter((n, n))
        constraints = [self._w <= w_max]
        if long_only:
            constraints += [self._w >= 0, cp_sum(self._w) <= gross_max]
        else:
            constraints += [norm1(self._w) <= gross_max]
        for sector, cap in (sector_caps or {}).items():
            members = [i for i, s in enumerate(sectors or []) if s == sector]
            if members:
                constraints.append(cp_sum(self._w[members]) <= cap)
        self._problem = cp.Problem(cp.Maximize(self._mu @ self._w - sum_squares(self._risk @ self._w)), constraints)

    def solve(self, mu: np.ndarray, sigma: np.ndarray, c: float) -> Allocation:
        """
        Solve for one (mu, Sigma, c). All cash (w = 0) meets every constraint, so the problem is always feasible,
        and with a positive-definite Sigma its optimum is unique: a failure is numerical, not a sign of impossible
        constraints. If the first attempt fails, the solve is retried once with RETRY_JITTER x mean(diag Sigma) added
        to Sigma's diagonal. If that fails too, the allocation has no weights and carries the reason; no substitute
        portfolio is ever returned, since one would look like a recommendation.
        """
        first = self._attempt(mu, sigma, c)
        if isinstance(first, Allocation):
            return first
        jitter = RETRY_JITTER * float(np.mean(np.diag(sigma)))
        retry = self._attempt(mu, sigma + jitter * np.eye(self.n), c)
        if isinstance(retry, Allocation):
            return retry
        reason = f"{first}; retry with a regularized covariance: {retry}"
        logger.warning("optimizer failed", extra={"event": "optimizer_failed", "reason": reason, "assets": self.n})
        return Allocation(None, None, "failed", reason)

    def _attempt(self, mu: np.ndarray, sigma: np.ndarray, c: float) -> Allocation | str:
        """
        One solve. The solution is used if cvxpy reports it optimal (or optimal_inaccurate) and it meets every
        constraint within FEASIBILITY_TOL; otherwise the reason it can't be used is returned.
        """
        try:
            chol = np.linalg.cholesky(sigma)
            self._mu.value = np.asarray(mu, dtype=float)
            self._risk.value = np.sqrt(1.0 / (2.0 * c)) * chol.T
            self._problem.solve(solver=SOLVER)  # type: ignore[no-untyped-call]  # cvxpy's solve is untyped
        except Exception as exc:
            return f"optimizer error ({type(exc).__name__}: {exc})"
        status = self._problem.status
        if status not in ACCEPTED_STATUSES or self._w.value is None:
            return f"solver status '{status}' (not optimal)"
        violation = max(float(np.max(con.violation())) for con in self._problem.constraints)  # type: ignore[no-untyped-call]
        if not violation <= FEASIBILITY_TOL:  # "not <=" so a NaN violation is rejected too
            return f"the solution breaks a constraint by {violation:.2g}"
        weights = np.asarray(self._w.value, dtype=float)
        weights[np.abs(weights) < 1e-9] = 0.0  # clear solver round-off so excluded assets show exactly 0
        return Allocation(weights, 1.0 - float(weights.sum()), str(status), None)


def constrained_kelly(
    mu: np.ndarray,
    sigma: np.ndarray,
    c: float,
    long_only: bool = True,
    gross_max: float = 1.0,
    w_max: float = 0.25,
    sector_caps: dict[str, float] | None = None,
    sectors: list[str | None] | None = None,
) -> Allocation:
    """One-off constrained fractional Kelly; see ConstrainedKelly for the problem and how failures are handled."""
    return ConstrainedKelly(len(mu), long_only, gross_max, w_max, sector_caps, sectors).solve(mu, sigma, c)


@dataclass(frozen=True)
class HRPResult:
    """HRP weights by ticker, the dendrogram leaf order and scipy's linkage matrix (for the cluster-tree chart)."""

    weights: pd.Series
    order: list[str]
    """Tickers in quasi-diagonal (dendrogram leaf) order."""
    linkage: np.ndarray


def hrp_weights(returns: pd.DataFrame) -> HRPResult:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016): correlation distance d = sqrt(0.5 (1 - rho)), single-linkage
    clustering, quasi-diagonalization (dendrogram leaf order), then recursive bisection: each split gets weight in
    inverse proportion to its cluster variance (computed with inverse-variance weights inside the cluster).
    HRP uses no expected-return estimate, which is a defensible choice given how noisy mean estimates are.
    """
    clean = returns.dropna()
    cov = clean.cov().to_numpy()
    corr = np.clip(clean.corr().to_numpy(), -1.0, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="single")
    order = [int(i) for i in leaves_list(link)]
    weights = np.ones(len(order))  # indexed by column position, like cov

    def cluster_variance(items: list[int]) -> float:
        """Variance of the inverse-variance-weighted portfolio of the assets at these column positions."""
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        var_left, var_right = cluster_variance(left), cluster_variance(right)
        alpha = 1 - var_left / (var_left + var_right)
        weights[left] *= alpha
        weights[right] *= 1 - alpha
        stack += [left, right]
    tickers = [str(c) for c in clean.columns]
    return HRPResult(pd.Series(weights, index=tickers), [tickers[i] for i in order], link)


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, solver: ConstrainedKelly, points: int = FRONTIER_POINTS
) -> pd.DataFrame:
    """
    Long-only constrained frontier traced by the Kelly multiplier c on a log grid (0.01 to 10): annualized volatility
    sqrt(w'Sigma w) and expected excess return w'mu of each optimal portfolio (cash earns rf, so 0 excess).
    Points where the optimizer fails are left out; the columns exist even if every point fails.
    """
    rows = []
    for c in np.logspace(-2, 1, points):
        w = solver.solve(mu, sigma, float(c)).weights
        if w is None:
            continue
        rows.append({"c": float(c), "volatility": float(np.sqrt(w @ sigma @ w)), "excess_return": float(w @ mu)})
    return pd.DataFrame(rows, columns=["c", "volatility", "excess_return"])
