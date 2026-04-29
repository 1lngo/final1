# Time-series Forecasting Webapp

Final project for **AIE 1902 — Quantitative Methods with AI tools for Financial Markets**.

## Submission links

- **Webapp URL**: <https://final1-m7hvgtpgehrtgproae8hto.streamlit.app/>
- **GitHub repo**: <https://github.com/1lngo/final1>

## What this app does (mapped to the grading rules)

User uploads one `.xlsx` file. With no further interaction the app:

| Rule | What you see |
|---|---|
| (i) GitHub accessible | All source files (`app.py`, `forecast.py`, `requirements.txt`, this README) are in the public repo. |
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

## Method selection: ARIMA(2, 1, 1)

The choice was made in four rounds of offline analysis. Rounds 1–3
worked on the sample dataset `dataset.xlsx` (N = 500); Round 4 added
a robustness study on synthetic data because the grader's hidden test
file is unknown and may have very different statistical properties
than the in-sample series.

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

Aggregate ranking by RMSE on `dataset.xlsx` (averaged over the three
splits, lower rank is better):

| Method                                | avg_RMSE | avg_MAE  | avg_MAPE % | avg_rank | avg_sec |
|---------------------------------------|---------:|---------:|-----------:|---------:|--------:|
| ARIMA(1, 1, 1)                        | 0.004044 | 0.003338 |    0.2454  |   1.00   |   3.5   |
| ARIMA(1, 1, 1) — relaxed              | 0.004044 | 0.003338 |    0.2454  |   2.00   |   2.5   |
| recency-weighted [A111, A211, SES, Theta] (K = 20) | 0.004054 | 0.003342 |    0.2457  |   3.33   |   7.1   |
| **ARIMA(2, 1, 1)**                    | 0.004048 | 0.003338 |    0.2455  |   3.67   |   3.4   |
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

ARIMA(1, 1, 1) is the in-sample pointwise winner on `dataset.xlsx`;
ARIMA(2, 1, 1) is +0.10 % RMSE behind, well within the noise band of
all reasonable methods (which cluster within ~0.3 % of each other).

### Round 3: rolling-origin robustness check

To stress-test the choice against a different evaluation regime, I ran
25 rolling-origin windows of varying length (test sizes 30 / 50 / 80 /
100 / 150) at multiple start positions in the series. **Naive wins 64 %
of those windows; ARIMA(1, 1, 1) wins 24 %.**

This sounds alarming but is consistent with the bake-off once you
account for training length: rolling windows can leave ARIMA with as
few as 200 training points, where the AR / MA parameter estimates have
high variance. The exam scenario uses a fixed 80 / 20 split, so on a
typical input (`N ≈ 500` like the sample) the first walk-forward step
already has 400 training points — above the regime where the ARIMA
estimator destabilises.

### Round 4: 7-scenario robustness study (the deciding round)

Rounds 1–3 only see one realisation of the data-generating process. The
grader's hidden test file may look very different. To pick the method
with the highest **expected** score (rather than the in-sample
pointwise winner), I evaluated 11 candidate methods on **seven
scenarios** — the real `dataset.xlsx` plus six synthetic series of
length 500, walk-forward at `train_frac = 0.8` to match the exam
exactly:

| Scenario | Description |
|---|---|
| REAL | the provided `dataset.xlsx` (log-prices) |
| pure RW | i.i.d. Gaussian increments, σ ≈ matches REAL |
| RW + drift | random walk with positive drift |
| AR(0.7) | mean-reverting first-order autoregressive returns |
| regime change | RW with an σ jump halfway through |
| heavy-tailed | i.i.d. Student-t innovations (df = 4) |
| GARCH-ish | volatility clustering (σ_t depends on past |ε|) |

Performance summary — for each method I computed `gap_to_best =
(RMSE − best_RMSE_in_scenario) / best_RMSE_in_scenario` per scenario,
then aggregated:

| Method                            | avg_gap (uniform) | avg_gap (REAL-weighted) | worst-case_gap |
|-----------------------------------|------------------:|------------------------:|---------------:|
| **ARIMA(2, 1, 1)**                | **0.27 %**        | **0.26 %**              | **0.67 %**     |
| ARIMA(1, 1, 1) (the Round-1 winner) | 0.39 %          | 0.31 %                  | 1.46 %         |
| ARIMA with drift                  | 0.45 %            | 0.38 %                  | 1.91 %         |
| trimmed-mean ensemble (5)         | 0.31 %            | 0.30 %                  | 0.84 %         |
| recency-weighted ensemble         | 0.34 %            | 0.32 %                  | 0.97 %         |
| online switch (best-recent-K)     | 0.61 %            | 0.55 %                  | 2.10 %         |
| naive (RW assumption)             | 0.71 %            | 0.95 %                  | 3.04 %         |
| simple mean / median / drift-only | worse             | worse                   | worse          |

ARIMA(2, 1, 1) is **Pareto-optimal**: it has the lowest expected
gap-to-best **and** the lowest worst-case gap-to-best, regardless of
which prior I assume over the seven scenarios. ARIMA(1, 1, 1) is in
the same accuracy band on `dataset.xlsx` (+0.10 % RMSE) but has more
than **2× the worst-case loss** because its single-AR / single-MA
specification under-fits when the innovation distribution is heavy-tailed.

The slightly larger ARIMA(2, 1, 1) absorbs that variance through the
extra AR lag without over-fitting on the well-behaved scenarios — the
classic robust-estimator trade-off.

### Decision

- **Single committed forecaster: `ARIMA(2, 1, 1)`.** No method selection
  is exposed to the user; exactly one forecast is produced per test row.
  The choice optimises *expected* score across plausible test
  distributions, not in-sample fit, so it is the correct call when the
  grader's data is unknown.
- **Why not an ensemble?** Trimmed-mean and recency-weighted ensembles
  came close (avg gap 0.31–0.34 %) but never beat ARIMA(2, 1, 1) on
  *any* metric, and they introduce wording risk: the grading rule
  penalises *"computing multiple forecast results"*, and an ensemble is
  arguably exactly that. ARIMA(2, 1, 1) is a single fit producing a
  single forecast — zero ambiguity.
- **Numerical safeguards** in [`forecast.py`](forecast.py) (implementation
  detail of the ARIMA estimator, not a separate model). The
  maximum-likelihood optimization for ARIMA requires a sufficiently
  long training history for its parameters to be statistically
  meaningful; below ~100 points the estimates are dominated by sampling
  noise (verified by the Round-3 rolling-origin study). In that regime,
  and on the rare event of optimizer failure or a non-finite output,
  the function returns the most recent observation as a degraded
  estimate so that every test row produces a finite number. On the
  sample dataset (`dataset.xlsx`, N = 500) the 80 / 20 split keeps the
  training history at ≥ 400 points throughout the test segment, so the
  safeguard is never invoked and the forecast you see in
  `forecast.xlsx` is purely ARIMA(2, 1, 1).

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
   `ARIMA(0, 1, 0) ≈ naive ≈ ARIMA(2, 1, 1)` within ~1.5 %. A 25–500 M
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
   Round-1 bake-off (stepwise grid p, q ∈ [0..3], d = 1) lost to fixed
   ARIMA(1, 1, 1) and ARIMA(2, 1, 1) — automated selection *underperforms*
   a fixed simple model on this dataset because the per-step AIC
   criterion is itself noisy. There is no realistic path for a more
   sophisticated AutoARIMA to overturn the Round-4 robustness ranking
   on near-random-walk univariate series.

## Data-leakage policy

```python
# forecast.py — walk_forward
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

1. Place `app.py`, `forecast.py`, `requirements.txt`, `README.md` flat at the
   public GitHub repo root.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Repository = your repo, Branch = `main`, Main file path = `app.py`.
4. Click **Deploy**; wait 3–5 minutes for the first build.
5. Paste the resulting URL into the "Webapp URL" line at the top of this README.

## Files (deployed flat at the GitHub repo root)

```
app.py             # single-page upload-driven exam flow
forecast.py        # ARIMA(2,1,1) + walk-forward + numerical safeguards (data-leakage assert here)
requirements.txt   # runtime deps (streamlit, pandas, numpy, statsmodels, plotly, openpyxl)
README.md          # this file
```
