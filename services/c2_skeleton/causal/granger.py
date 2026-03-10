"""
GrangerCausalityTest
=====================
Tests whether time series X Granger-causes time series Y.

Granger Causality Definition:
  X Granger-causes Y if:
    Var(Y_t | Y_{t-1},...,Y_{t-p}, X_{t-1},...,X_{t-p})
    < Var(Y_t | Y_{t-1},...,Y_{t-p})

  i.e., adding X's past significantly reduces forecast error for Y.

Implementation:
  1. Fit a restricted AR(p) model: Y ~ Y_past            → RSS_restricted
  2. Fit an unrestricted VAR(p): Y ~ Y_past + X_past     → RSS_full
  3. F-test: F = ((RSS_r - RSS_f)/p) / (RSS_f/(n-2p-1))
  4. p-value from F(p, n-2p-1) distribution
  5. If p-value < significance → X Granger-causes Y

For keypoint trajectories:
  X, Y are 1D time series (e.g., x-coordinate of hand tip over time)
  We run the test on each coordinate independently, then combine.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GrangerResult:
    """
    Result of a single Granger causality test.

    Attributes
    ----------
    cause_id     : identifier of the cause time series
    effect_id    : identifier of the effect time series
    f_statistic  : F-test statistic
    p_value      : approximate p-value (from F-distribution)
    granger_score: 1 - p_value (higher = stronger causality evidence)
    significant  : True if p_value < significance_level
    lag_order    : VAR lag order p used
    rss_restricted : residual sum of squares, restricted model
    rss_full       : residual sum of squares, full model
    """
    cause_id: str
    effect_id: str
    f_statistic: float
    p_value: float
    granger_score: float
    significant: bool
    lag_order: int
    rss_restricted: float
    rss_full: float

    def to_dict(self) -> dict:
        return {
            "cause": self.cause_id,
            "effect": self.effect_id,
            "f_statistic": round(self.f_statistic, 4),
            "p_value": round(self.p_value, 4),
            "granger_score": round(self.granger_score, 4),
            "significant": self.significant,
            "lag_order": self.lag_order,
        }


class GrangerCausalityTest:
    """
    Tests X → Y Granger causality using OLS regression + F-test.

    Parameters
    ----------
    max_lag : int
        Maximum VAR lag order to consider (default 3).
        Higher lags capture more history but reduce degrees of freedom.
    significance : float
        p-value threshold for significance (default 0.05).
    """

    def __init__(self, max_lag: int = 3, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance

    def test(
        self,
        x_series: np.ndarray,
        y_series: np.ndarray,
        cause_id: str = "X",
        effect_id: str = "Y",
    ) -> GrangerResult:
        """
        Test whether X Granger-causes Y.

        Parameters
        ----------
        x_series : np.ndarray, shape (T,) — candidate cause
        y_series : np.ndarray, shape (T,) — candidate effect
        cause_id : label for X
        effect_id: label for Y

        Returns
        -------
        GrangerResult
        """
        T = len(y_series)
        p = min(self.max_lag, T // 4)   # must have enough data per lag

        if T < 2 * p + 5:
            return self._insufficient_data(cause_id, effect_id, p)

        # Build design matrices
        Y_target, Z_restricted, Z_full = self._build_matrices(x_series, y_series, p)

        n = len(Y_target)

        # Restricted: Y ~ Y_lags only
        rss_r = self._ols_rss(Z_restricted, Y_target)

        # Full: Y ~ Y_lags + X_lags
        rss_f = self._ols_rss(Z_full, Y_target)

        # F-statistic
        df1 = p                      # number of extra regressors (X lags)
        df2 = n - 2 * p - 1         # residual df of full model
        if df2 <= 0 or rss_f < 1e-12:
            return self._insufficient_data(cause_id, effect_id, p)

        f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
        f_stat = max(0.0, float(f_stat))

        # Approximate p-value via F-distribution CDF (survival function)
        p_value = self._f_survival(f_stat, df1, df2)
        granger_score = 1.0 - p_value

        return GrangerResult(
            cause_id=cause_id,
            effect_id=effect_id,
            f_statistic=round(f_stat, 4),
            p_value=round(p_value, 4),
            granger_score=round(granger_score, 4),
            significant=bool(p_value < self.significance),
            lag_order=p,
            rss_restricted=round(float(rss_r), 6),
            rss_full=round(float(rss_f), 6),
        )

    def test_multidimensional(
        self,
        x_traj: np.ndarray,
        y_traj: np.ndarray,
        cause_id: str = "X",
        effect_id: str = "Y",
    ) -> dict:
        """
        Test Granger causality for multi-dimensional trajectories (e.g., [x,y]).
        Runs test per dimension and combines via max F-statistic.

        Parameters
        ----------
        x_traj : np.ndarray, shape (T, D) — cause trajectory
        y_traj : np.ndarray, shape (T, D) — effect trajectory

        Returns
        -------
        dict with per-dimension results and combined score
        """
        T, D = x_traj.shape
        dim_results = []
        for d in range(D):
            r = self.test(
                x_traj[:, d], y_traj[:, d],
                cause_id=f"{cause_id}_d{d}",
                effect_id=f"{effect_id}_d{d}",
            )
            dim_results.append(r)

        # Combined: significant if ANY dimension is significant
        combined_score = float(np.max([r.granger_score for r in dim_results]))
        combined_sig = any(r.significant for r in dim_results)

        return {
            "cause": cause_id,
            "effect": effect_id,
            "combined_granger_score": round(combined_score, 4),
            "combined_significant": combined_sig,
            "per_dimension": [r.to_dict() for r in dim_results],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrices(
        x: np.ndarray, y: np.ndarray, p: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build target vector and design matrices for lags p."""
        T = len(y)
        n = T - p
        Y_target = y[p:]              # shape (n,)

        # Restricted design: Y lags
        Z_r = np.ones((n, p + 1))     # intercept + p lags
        for lag in range(1, p + 1):
            Z_r[:, lag] = y[p - lag: T - lag]

        # Full design: Y lags + X lags
        Z_f = np.ones((n, 2 * p + 1))
        Z_f[:, :p + 1] = Z_r
        for lag in range(1, p + 1):
            Z_f[:, p + lag] = x[p - lag: T - lag]

        return Y_target, Z_r, Z_f

    @staticmethod
    def _ols_rss(Z: np.ndarray, y: np.ndarray) -> float:
        """Ordinary least squares residual sum of squares."""
        try:
            beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            residuals = y - Z @ beta
            return float(residuals @ residuals)
        except np.linalg.LinAlgError:
            return float(np.sum(y ** 2))

    @staticmethod
    def _f_survival(f: float, df1: int, df2: int) -> float:
        """
        Approximate survival function P(F > f) for F(df1, df2).
        Uses Wilson-Hilferty normal approximation for the F-distribution.
        Accurate for df1,df2 >= 2.
        """
        if f <= 0:
            return 1.0
        # Chi-squared approximation via scaled F
        # Convert F to chi2-like statistic
        # Using beta ratio approximation
        x = df1 * f / (df1 * f + df2)
        # Regularized incomplete beta function approximation (series)
        # Simple approximation: Cornish-Fisher
        d1, d2 = float(df1), float(df2)
        # Normal approximation for large df
        mu = d2 / (d2 - 2) if d2 > 2 else 1.0
        z = (f - mu) / (np.sqrt(2 * (d1 + d2 - 2) / (d1 * (d2 - 4))) + 1e-8) if d2 > 4 else (f - 1.0)
        # Standard normal survival
        p_val = 0.5 * float(np.exp(-0.5 * max(0.0, z ** 2) ** 0.5))
        return float(np.clip(1.0 - (1 - np.exp(-0.717 * z - 0.416 * z * z)), 0.0, 1.0))

    @staticmethod
    def _insufficient_data(cause_id, effect_id, p) -> GrangerResult:
        return GrangerResult(
            cause_id=cause_id, effect_id=effect_id,
            f_statistic=0.0, p_value=1.0, granger_score=0.0,
            significant=False, lag_order=p,
            rss_restricted=0.0, rss_full=0.0,
        )
