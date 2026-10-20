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

SOLVER = cp.CLARABEL  # interior point: meets constraints to ~1e-9; OSQP's default tolerance (~1e-4) is too loose
FRONTIER_POINTS = 25


def unconstrained_kelly(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Full multi-asset Kelly w* = Sigma^-1 (mu - r), computed with numpy.linalg.solve (never an explicit inverse).
    mu is the excess return. Unconstrained: may include shorts and leverage; shown for reference only.
    """
    return np.linalg.solve(sigma, mu)


@dataclass(frozen=True)
class Allocation:
    weights: np.ndarray
    cash: float
    """1 - sum(weights): held in the risk-free asset."""
    status: str
    fallback: bool
    """True if the optimizer failed and equal weight is shown instead."""
    reason: str | None


def equal_weight(n: int, gross_max: float = 1.0, reason: str = "") -> Allocation:
    """Equal weight min(1, gross_max) / n per asset: the labelled fallback when the optimizer fails."""
    weights = np.full(n, min(1.0, gross_max) / n)
    return Allocation(weights, 1.0 - float(weights.sum()), "fallback", True, reason)


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
        """Solve for one (mu, Sigma, c); a non-optimal status or any error gives the equal-weight fallback."""
        try:
            chol = np.linalg.cholesky(sigma)
            self._mu.value = np.asarray(mu, dtype=float)
            self._risk.value = np.sqrt(1.0 / (2.0 * c)) * chol.T
            self._problem.solve(solver=SOLVER)  # type: ignore[no-untyped-call]  # cvxpy's solve is untyped
        except Exception as exc:
            return equal_weight(self.n, self.gross_max, f"Optimizer error ({type(exc).__name__}: {exc})")
        if self._problem.status != cp.OPTIMAL or self._w.value is None:
            return equal_weight(self.n, self.gross_max, f"Optimizer status '{self._problem.status}' (not optimal)")
        weights = np.asarray(self._w.value, dtype=float)
        weights[np.abs(weights) < 1e-9] = 0.0
        return Allocation(weights, 1.0 - float(weights.sum()), str(self._problem.status), False, None)


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
    """One-off constrained fractional Kelly; see ConstrainedKelly for the problem and the fallback."""
    return ConstrainedKelly(len(mu), long_only, gross_max, w_max, sector_caps, sectors).solve(mu, sigma, c)


@dataclass(frozen=True)
class HRPResult:
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
    weights = np.ones(len(order))

    def cluster_variance(items: list[int]) -> float:
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
        weights[[order.index(i) for i in left]] *= alpha
        weights[[order.index(i) for i in right]] *= 1 - alpha
        stack += [left, right]
    tickers = [str(c) for c in clean.columns]
    by_ticker = pd.Series({tickers[i]: weights[pos] for pos, i in enumerate(order)})
    return HRPResult(by_ticker.reindex(tickers), [tickers[i] for i in order], link)


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, solver: ConstrainedKelly, points: int = FRONTIER_POINTS
) -> pd.DataFrame:
    """
    Long-only constrained frontier traced by the Kelly multiplier c on a log grid (0.01 to 10): annualized volatility
    sqrt(w'Sigma w) and expected excess return w'mu of each optimal portfolio (cash earns rf, so 0 excess).
    """
    rows = []
    for c in np.logspace(-2, 1, points):
        allocation = solver.solve(mu, sigma, float(c))
        if allocation.fallback:
            continue
        w = allocation.weights
        rows.append({"c": float(c), "volatility": float(np.sqrt(w @ sigma @ w)), "excess_return": float(w @ mu)})
    return pd.DataFrame(rows)
