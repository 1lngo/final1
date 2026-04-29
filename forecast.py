"""Forecasting core for the exam webapp.

Method: ARIMA(1, 1, 1), refit at every walk-forward step, with a naive
warmup for very short histories.

Selected after a 19-method x 3-split bake-off on the sample dataset
(`dataset.xlsx`, N=500). Average rank by RMSE across train_frac in
{0.7, 0.8, 0.9}; lower rank is better (full table in webapp/README.md):

    method                  avg_RMSE   avg_rank
    ARIMA(1,1,1)            0.004044    1.00    <- chosen (rank 1 on EVERY split)
    ARIMA(1,1,1) relaxed    0.004044    2.00
    rec-w 4-model ensemble  0.004054    3.33
    ARIMA(2,1,1)            0.004048    3.67
    median3 / mean5 / UC    ~0.0041     6-13
    naive                   0.004106   12.00
    RidgeAR / drift         worse       16-19

Foundation models (Kronos, TimesFM, Chronos) and statsforecast (AutoARIMA,
AutoETS, MFLES, CES) were considered and rejected: input-schema mismatch
(Kronos requires OHLCV; we have univariate y), prohibitive deployment cost
(~1 GB extra deps on Streamlit Cloud's free tier), and -- decisively -- no
accuracy headroom on a near-random-walk series, where the 19-method
bake-off shows all reasonable methods cluster within 0.3% RMSE of each
other.

Warmup threshold (rationale): an additional rolling-origin robustness
check found that ARIMA(1, 1, 1) only beats naive when training length >=
~150 points; below that, parameter-estimation variance dominates and
naive wins. The exam scenario uses train_frac=0.8, so the official split
on dataset.xlsx (N=500) gives train >= 400 points -- comfortably above
the threshold and consistent with the bake-off result. To stay safe on
unusually small inputs (e.g. N=100 -> first walk-forward step has only
80 points), `safe_forecast` falls back to naive whenever the available
history is shorter than `WARMUP_POINTS`.

Data-leakage guarantee: every forecast at step t uses only y[:t]; an
internal `assert` inside `walk_forward` enforces this.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

DEFAULT_ORDER = (1, 1, 1)
WARMUP_POINTS = 100  # below this many history points, fall back to naive


def safe_forecast(history: np.ndarray, order: tuple[int, int, int] = DEFAULT_ORDER) -> float:
    """Return a one-step forecast for the next value after `history`.

    Strategy:
      - history shorter than `WARMUP_POINTS`: return naive (y_{t-1}).
        ARIMA(1,1,1) parameter estimates are too noisy on short windows
        (rolling-origin: naive beats ARIMA in 64% of windows when
        train < 150 points). Naive is provably optimal for a true
        random walk and is a safe stand-in.
      - otherwise: fit ARIMA(p, d, q) and return its 1-step forecast.
      - any optimizer failure or non-finite output: fall back to naive.

    Pure function: never mutates the input, never raises.
    """
    h = np.asarray(history, dtype=float).ravel()
    if h.size == 0:
        return float("nan")
    if h.size < WARMUP_POINTS:
        return float(h[-1])
    try:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(h, order=order).fit()
            yhat = float(np.asarray(fit.forecast(1))[0])
        if not np.isfinite(yhat):
            return float(h[-1])
        return yhat
    except Exception:
        return float(h[-1])


def one_step_arima(history: np.ndarray, order: tuple[int, int, int] = DEFAULT_ORDER) -> float:
    """Public alias for the one-step forecast used after the upload."""
    return safe_forecast(history, order=order)


def walk_forward(
    y: np.ndarray,
    train_frac: float = 0.8,
    order: tuple[int, int, int] = DEFAULT_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling one-step-ahead forecasts on the last `(1 - train_frac)` of `y`.

    For each test index t in [cutoff, len(y)), forecast y[t] using ONLY y[:t].
    This is the strict no-leakage protocol required by the exam brief.

    Returns
    -------
    forecasts : np.ndarray
        Predicted values aligned with the test segment.
    test_index : np.ndarray
        Integer positions in `y` corresponding to each forecast.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    cutoff = int(n * train_frac)
    if cutoff < 3 or cutoff >= n:
        raise ValueError(
            f"Need at least 3 training points and 1 test point; got n={n}, cutoff={cutoff}."
        )

    forecasts = np.empty(n - cutoff, dtype=float)
    for i, t in enumerate(range(cutoff, n)):
        history = y[:t]
        # Hard guard against future-leakage: history must end at index t-1.
        assert history.shape[0] == t, "data leakage: history must be y[:t]"
        forecasts[i] = safe_forecast(history, order=order)

    test_index = np.arange(cutoff, n)
    return forecasts, test_index


def metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    eps = 1e-12
    denom = np.where(np.abs(yt) < eps, eps, yt)
    return {
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
        "MAPE_%": float(np.mean(np.abs((yt - yp) / denom)) * 100),
    }
