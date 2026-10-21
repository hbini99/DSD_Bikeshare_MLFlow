# Capital Bikeshare next-day forecast — MLflow project

Two analysts test two forecasting approaches for **tomorrow's total bike rentals**. Every training
run is tracked in MLflow with its data period, input features and settings. The best model is
registered under a name and only becomes the **champion** after it beats the planning rule
operations already uses: *tomorrow's rentals = today's rentals*.

## The problem

- **Who needs it.** Capital Bikeshare's operations team plans staffing and fleet capacity each evening.
- **The cost of getting it wrong.** Too low leaves the team unprepared for a busy day. Too high
  wastes operating cost.
- **The constraint.** The forecast is made the evening before, so it may only use information
  available at that point.

| | |
|---|---|
| Data | UCI Bike Sharing daily records, 2011-01-01 to 2012-12-31 (731 days), `data/day.csv` |
| Target | `cnt`: total rentals on the next day |
| Benchmark | Naive rule: tomorrow = today's `cnt` |
| Metric | Mean absolute error (MAE) in rentals/day; skill = 1 − MAE_model / MAE_naive |

**What "available the evening before" means.** A forecast for day *d* uses:

- rentals up to day *d-1* (yesterday's total, same weekday last week, 7- and 28-day averages, casual share);
- the weather observed on day *d-1*;
- day *d*'s calendar (weekday, working day, holiday).

The dataset's weather columns describe conditions *during* each day, so day *d*'s own weather is
never used. Neither are day *d*'s casual and registered counts, which add up to the target.

## Project structure

```
data/day.csv              UCI Bike Sharing daily data
common.py                 features, time-based splits, metrics, and the logged MLflow run
train_ridge.py            Approach A (analyst A): Ridge regression, one run per alpha
train_gbm.py              Approach B (analyst B): gradient-boosted trees, one run per setting
register_champion.py      rank runs -> register best -> benchmark check -> @champion alias
verify_champion.py        load the champion by name, trace its lineage, re-check its results
requirements.txt
```

## Setup

Python 3.12:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Start tracking the training experiments

**Option 1 — local file (default, nothing else to start).** The scripts write to `mlflow.db`
in the project folder. Open the UI in a second terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # http://127.0.0.1:5000
```

**Option 2 — tracking server** (what a team would share). Start the server, then point the
scripts at it:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000                   # Windows: set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Then run the two approaches and the registration:

```bash
python train_ridge.py          # 4 runs: alpha = 1, 3, 10, 30
python train_gbm.py            # 3 runs: learning rate / tree size / iterations
python register_champion.py    # registers the best run, benchmark check, @champion alias
python verify_champion.py      # retrieves the champion by name and re-checks it
```

`train_ridge.py --alphas 0.3 1 3` runs a different grid. Each run appears in the same
experiment, `bikeshare-next-day-forecast`.

## What each run logs

| MLflow field | Content | Why |
|---|---|---|
| Tags | `approach`, `analyst`, `data_sha256` (fingerprint of `day.csv`), `git_commit`, `model_uri` | Who ran it, with which exact data file and code version |
| Parameters | `train_start/end`, `val_start/end`, row counts, `features`, `target`, model type and hyperparameters | Which data period, inputs and settings produced the score |
| Datasets | Training and validation tables (`mlflow.log_input`) | The data itself is linked to the run, not just described |
| Metrics | `val_mae`, `val_rmse`, `val_mape`, `val_naive_mae`, `val_skill_vs_naive`, `val_under_forecast_share`, `val_bias`, `train_mae` | Error scores side by side with the benchmark on the same days |
| Artifacts | `evaluation/val_predictions.csv`, `evaluation/val_forecast.png`, `model/` (with signature and input example) | Day-by-day evidence, and the model itself |

## How the champion is chosen

| Period | Dates | Used for |
|---|---|---|
| Warm-up | 2011-01-01 to 2011-01-28 | Building the 28-day averages only |
| Train | 2011-01-29 to 2012-06-30 | Fitting every run |
| Validation | 2012-07-01 to 2012-09-30 | Comparing runs across both approaches |
| Test | 2012-10-01 to 2012-12-31 | The benchmark check only, never used to choose |

`register_champion.py` follows these steps:

1. Ranks all runs from both approaches by validation MAE.
2. Registers the best as a version of **`bikeshare-next-day-rentals`**. The version keeps a link
   to its source run and is tagged with `source_run_id`, `approach`, `analyst`, `data_sha256`
   and `val_mae`. Re-running the script re-checks the same version instead of creating a duplicate.
3. Loads that version **by name** and scores the untouched test period against the naive rule.
   The check is logged as its own run, `champion-check-vN`.
4. **Only if it beats the naive rule** does it get the alias **`@champion`**. Operations retrieves
   the model with `models:/bikeshare-next-day-rentals@champion`. If it loses, it is tagged
   `benchmark_check=failed` and any previous champion stays in place.

## Verify the results

**In the MLflow UI** (http://127.0.0.1:5000):

1. **Experiments → bikeshare-next-day-forecast.**
   - Add the columns `approach`, `val_mae`, `val_naive_mae` and `val_skill_vs_naive`.
   - Sort by `val_mae`.
   - Select runs and click **Compare** to see the parameters and scores side by side.
2. **Open any run.**
   - *Parameters* show the data period, features and settings.
   - *Tags* show the data fingerprint and git commit.
   - *Datasets used* shows the training and validation data.
   - *Artifacts* hold the validation forecast plot and the logged model.
3. **Models → bikeshare-next-day-rentals.**
   - The champion version carries the `@champion` alias and the tags `benchmark_check=passed`,
     `test_mae` and `test_naive_mae`.
   - Its *Source run* link opens the training run that produced it.

**From the command line:** `python verify_champion.py` does the following.

- Loads `@champion` by name.
- Prints the run, approach, data period, data fingerprint and settings behind it.
- Re-scores the test period and asserts the MAE matches the value recorded at registration.
- Shows the last 7 days of forecasts against actuals and the naive rule.

## Results

| Approach | Run | Validation MAE | Naive MAE | Skill |
|---|---|---|---|---|
| A — Ridge | `ridge-alpha10` | **683** | 728 | **+6.3%** (champion) |
| A — Ridge | `ridge-alpha3` | 686 | 728 | +5.8% |
| A — Ridge | `ridge-alpha30` | 687 | 728 | +5.7% |
| A — Ridge | `ridge-alpha1` | 689 | 728 | +5.4% |
| B — Gradient boosting | `gbm-lr0.05-leaves7-it100` | 848 | 728 | −16.4% |
| B — Gradient boosting | `gbm-lr0.03-leaves7-it300` | 886 | 728 | −21.6% |
| B — Gradient boosting | `gbm-lr0.05-leaves15-it200` | 903 | 728 | −24.0% |

**Benchmark check** for `bikeshare-next-day-rentals` v1 (`ridge-alpha10`) on Oct–Dec 2012:
MAE **897 vs 952** for the naive rule (+5.7%) → **passed, alias `@champion`**.

### Interpretation

- **The margin over the naive rule is small, about 6%, and that is expected.** Tomorrow's weather
  drives most of the day-to-day swing and is not known the evening before. A model that beat the
  naive rule by a wide margin on this data would usually be using same-day weather, which is leakage.
- **Ridge beat gradient boosting** even though both predict the deviation from the last 7 days'
  average. With only 519 training days, the trees overfit. Their training MAE is 241–486 against
  848–903 on validation. Ridge's training and validation MAE are close: 594 vs 683 for α = 10.
  `train_mae` is logged on every run so this gap is visible in the UI.
- **The champion under-forecasts on 54% of test days.** The brief says under-forecasting is the
  costlier mistake, but MAE treats both directions equally. A quantile loss or a safety buffer is
  the natural next step.
- **Holidays are the weak spot.** On 25 Dec 2012 the champion forecast 2,723 rentals against
  1,013 actual. There are only 14 holidays in the training period, too few for the model to
  learn how much they depress demand. The test period
  also contains Hurricane Sandy (29 Oct 2012, 22 rentals), which no forecast from yesterday's
  data can anticipate.
- **The best improvement would be a weather forecast for tomorrow.** It is available in the
  evening, so it is a legitimate input, and it targets the main source of error.

## Reproducibility

- **Pinned dependencies.** Versions are fixed in `requirements.txt`.
- **Fixed splits and seeds.** The periods are fixed in `common.py`, and gradient boosting uses `random_state=42`.
- **Same inputs, same outputs.** Re-running the four commands on the same `day.csv` (same
  `data_sha256`) reproduces the table above.
- **Generated files stay out of git.** `mlflow.db` and `mlruns/` are created by the scripts and
  excluded by `.gitignore`.

Data source: Fanaee-T, H. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository.
The operational scenario is illustrative.
