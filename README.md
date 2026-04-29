# Time-series Forecasting Webapp

Final project for **AIE 1902 — Quantitative Methods with AI tools for Financial Markets**.

## Submission links

- **Webapp URL**: `<paste Streamlit Cloud URL here after deploy>`
- **GitHub repo**: `<paste GitHub URL here>`

## What this app does (mapped to the grading rules)

User uploads one `.xlsx` file. With no further interaction the app:

| Rule | What you see |
|---|---|
| (i) GitHub accessible | All source files (`app.py`, `requirements.txt`, `utils/`, this README) are in the public repo. |
| (ii) Webapp accessible | Single-page Streamlit app deployed to Streamlit Community Cloud. |
| (iii) One-step forecast | "Part 1" header shows `y_{N+1}` immediately after upload; the user does **not** pick a method. |
| (iv) Backtest | "Part 2" header runs a walk-forward backtest over the last 20% of the series, reporting RMSE / MAE / MAPE and an actual-vs-forecast plot. |
| (v) Excel output | A `Download forecast.xlsx` button, with locked column order and asserted row count. |
| Data-leakage 0-pt rule | `walk_forward` slices `y[:t]` for every test index `t`, with an `assert` enforcing it. No future observations enter any fit. |

## Input schema

| Column | Required | Notes |
|---|---|---|
| `y` | yes | Numeric, ordered oldest → newest, **no missing values** |
| `date` | optional | Used only for display / output alignment |

The first sheet of the workbook is used regardless of name.

The app rejects files that fail any of these checks (missing `y`, NaN in `y`,
non-numeric `y`, `N < 10`) with a friendly red message.

## Output schema (`forecast.xlsx`)

- Sheet name: `forecast`
- Columns: `[date, y]` if the input had a `date` column, else `[y]`
- Rows: exactly `N - floor(0.8 * N)` (the walk-forward test segment, in
  chronological order)
- Dtype of `y`: float
- Runtime asserts in `app.py` will fail loudly if any of the above ever
  regresses.

## Method selection: ARIMA(1, 1, 1) with naive warmup

The choice was made in three rounds of offline analysis on the sample
dataset `dataset.xlsx` (N = 500).

### Round 1: 14-method bake-off (initial sweep)

Naive, drift, ARIMA orders (0,1,0) … (3,1,3), SES, Holt (additive /
damped), Theta, auto-ARIMA-AIC (stepwise on a (p, 1, q) grid with
p, q ∈ [0..3]), and a 3-model combination (ARIMA(1,1,1) + SES + Theta).
Walk-forward at `train_frac ∈ {0.7, 0.8, 0.9}`.

### Round 2: 19-method extended bake-off

Round 1 plus state-space `UnobservedComponents` (local level / local
linear trend / smooth trend), `ETSModel` (ANN, AAdN), Ridge-AR(p)
on differenced series for p ∈ {5, 10}, mean-of-5, median-of-5,
recency-weighted ensemble (weights ∝ 1 / RMSE over last K=20 forecasts).

Aggregate ranking by RMSE (averaged over the three splits, lower rank
is better):

| Method                                | avg_RMSE | avg_MAE  | avg_MAPE % | avg_rank | avg_sec |
|---------------------------------------|---------:|---------:|-----------:|---------:|--------:|
| **ARIMA(1, 1, 1)**                    | 0.004044 | 0.003338 |    0.2454  | **1.00** |   3.5   |
| ARIMA(1, 1, 1) — relaxed              | 0.004044 | 0.003338 |    0.2454  |   2.00   |   2.5   |
| recency-weighted [A111, A211, SES, Theta] (K = 20) | 0.004054 | 0.003342 |    0.2457  |   3.33   |   7.1   |
| ARIMA(2, 1, 1)                        | 0.004048 | 0.003338 |    0.2455  |   3.67   |   3.4   |
| recency-weighted [A111, SES, Theta]   | 0.004057 | 0.003343 |    0.2458  |   5.67   |   4.0   |
| mean-of-5                             | 0.004056 | 0.003342 |    0.2458  |   6.00   |   8.6   |
| median-3 (ARIMA + SES + Theta)        | 0.004064 | 0.003347 |    0.2461  |   8.33   |   3.9   |
| UnobservedComponents (local level)    | 0.004065 | 0.003347 |    0.2461  |  10.00   |   2.0   |
| SES                                   | 0.004065 | 0.003347 |    0.2461  |  10.33   |   0.2   |
| ETSModel ANN                          | 0.004065 | 0.003347 |    0.2461  |  11.67   |   0.4   |
| Theta                                 | 0.004065 | 0.003348 |    0.2462  |  12.67   |   0.9   |
| ETSModel AAdN                         | 0.004066 | 0.003348 |    0.2461  |  12.67   |   1.7   |
| naive                                 | 0.004106 | 0.003365 |    0.2474  |  12.00   |   0.0   |
| Ridge-AR(5/10), drift, smooth-trend   | worse    | worse    | worse      | 14–19    | varies  |

`ARIMA(1, 1, 1)` is rank 1 on **every** split (avg_rank 1.00), not just
on average — it is the pointwise-best method on this dataset.

### Round 3: rolling-origin robustness check

To stress-test the choice against a different evaluation regime, I ran
25 rolling-origin windows of varying length (test sizes 30 / 50 / 80 /
100 / 150) at multiple start positions in the series. **Naive wins 64 %
of those windows; ARIMA(1, 1, 1) wins 24 %.**

This sounds alarming but is consistent with the bake-off once you
account for training length: rolling windows can leave ARIMA with as
few as 200 training points, where the AR / MA parameter estimates have
high variance and naive (which is provably optimal for a pure random
walk) edges out. The exam scenario, however, uses a fixed 80 / 20 split,
so on a typical input (`N ≈ 500` like the sample) the first walk-forward
step already has 400 training points — well above the regime where
ARIMA(1, 1, 1) destabilises.

### Decision

- Primary forecaster: `ARIMA(1, 1, 1)`.
- **Naive warmup** in [`utils/forecast.py`](utils/forecast.py) — when the
  available history has fewer than `WARMUP_POINTS = 100` observations,
  return `y_{t-1}` instead of fitting ARIMA. On the sample dataset this
  threshold is never hit (the test segment starts at `t = 400`), so the
  bake-off result is preserved exactly. On unusually small inputs (e.g.
  N = 100, where the first test step would have only 80 points), the
  warmup gracefully degrades to naive — which the rolling-origin study
  showed is the genuinely better choice in that low-data regime.
- Naive fallback also covers any optimizer failure or non-finite output,
  guaranteeing 100% finite test-segment coverage in `forecast.xlsx`.

## What was considered and rejected

### Foundation TSFMs (Kronos, TimesFM, Chronos, Lag-Llama)

Three independent reasons:

1. **Schema mismatch.** Kronos and similar K-line foundation models are
   pre-trained on multivariate OHLCV inputs. The exam input is a single
   column `y` (a linear combination of log-close prices). Forcing
   `O = H = L = C = y, V = 0` puts the data far outside their training
   distribution — a textbook misuse that consistently underperforms simple
   baselines on near-random-walk univariate series.
2. **Deployment cost.** Adding `torch + transformers + huggingface_hub`
   plus a 100 MB-class checkpoint inflates the deployment footprint to
   ~1 GB, slows Streamlit Cloud cold-start by minutes, and risks OOM /
   build failure on the free 1 GB tier — directly threatening rules (i)
   and (ii).
3. **No accuracy headroom.** The bake-off above shows
   `ARIMA(0, 1, 0) ≈ naive ≈ ARIMA(1, 1, 1)` within ~1.5%. A 25–500 M
   parameter transformer cannot extract structure from a series that is
   essentially a random walk; the same logic excludes TimesFM, Chronos,
   and Lag-Llama.

### Nixtla `statsforecast` (AutoARIMA / AutoETS / MFLES / CES)

Even though `statsforecast`'s AutoARIMA is the most thorough open-source
order-search available, two reasons make it a net negative here:

1. **Install fragility.** It pulls a `scipy` version range that has no
   prebuilt wheel for Python 3.14 (the local runtime), forcing a from-source
   compile that needs a Visual Studio toolchain. On the Streamlit Cloud
   build (Python 3.12) wheels exist, but adding the dependency is still a
   ~30 MB install plus numba JIT cold-start cost on every container restart.
2. **No expected gain.** The hand-coded auto-ARIMA-AIC variant in the
   Round-1 bake-off (stepwise grid p, q ∈ [0..3], d = 1) lost to the fixed
   `ARIMA(1, 1, 1)` (avg_rank 8.33 vs 1.00) — automated selection
   *underperforms* a fixed simple model on this dataset because the
   per-step AIC criterion is itself noisy. There is no realistic path
   for a more sophisticated AutoARIMA to overturn this on a series with
   only ~1.5 % of total RMSE worth of extractable signal.

## Data-leakage policy

```python
# webapp/utils/forecast.py — walk_forward
for i, t in enumerate(range(cutoff, n)):
    history = y[:t]
    assert history.shape[0] == t, "data leakage: history must be y[:t]"
    forecasts[i] = safe_forecast(history, order=order)
```

Every prediction at test index `t` is fit on, and only on, `y[0..t-1]`.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501> and upload an `.xlsx`.

## Deploy on Streamlit Community Cloud

1. Push this `webapp/` directory to a public GitHub repo (root of the repo).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Repository = your repo, Branch = `main`, Main file path = `app.py`.
4. Click **Deploy**; wait 3–5 minutes for the first build.
5. Paste the resulting URL into the "Webapp URL" line at the top of this README.

## Files

```
app.py                 # single-page upload-driven exam flow
requirements.txt       # runtime deps (streamlit, pandas, numpy, statsmodels, plotly, openpyxl)
.streamlit/config.toml # theme + server options
.gitignore
utils/
  __init__.py
  forecast.py          # ARIMA(1,1,1) + walk-forward + naive fallback (data-leakage assert here)
README.md              # this file
```
