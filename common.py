"""Shared pieces for both forecasting approaches: data, leakage-safe features, time-based
splits, metrics and one function that performs a fully logged MLflow training run.

The forecast for TARGET day d is made on the evening of day d-1, so it may only use:
  * rentals and weather observed up to and including day d-1 (lagged features)
  * calendar facts about day d (weekday, working day, holiday), known in advance
Day-d weather and day-d casual/registered counts are never used: they are observed during
the day being forecast, so using them would be leakage.
"""
import hashlib
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin, clone

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "day.csv"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT = "bikeshare-next-day-forecast"
MODEL_NAME = "bikeshare-next-day-rentals"

# Time-based periods (inclusive). The test period is only used by register_champion.py.
WARMUP_DAYS = 28
TRAIN_END = "2012-06-30"
VAL_START, VAL_END = "2012-07-01", "2012-09-30"
TEST_START, TEST_END = "2012-10-01", "2012-12-31"

NUMERIC = ["lag1", "lag7", "roll7", "roll28", "lag1_rel", "lag1_casual_share",
           "temp_lag1", "hum_lag1", "windspeed_lag1",
           "workingday", "holiday", "wd_change", "doy_sin", "doy_cos"]
CATEGORICAL = ["weekday", "weathersit_lag1"]
FEATURES = NUMERIC + CATEGORICAL

warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


# ----------------------------------------------------------------------------- data
def file_sha256(path=DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
    s = raw["cnt"].astype(float)
    f = pd.DataFrame({"date": raw["dteday"], "y": s})
    f["lag1"] = s.shift(1)                          # also the naive benchmark forecast
    f["lag7"] = s.shift(7)                          # same weekday last week
    f["roll7"] = s.shift(1).rolling(7).mean()
    f["roll28"] = s.shift(1).rolling(28).mean()
    f["lag1_rel"] = f["lag1"] / f["roll7"]          # was today unusually busy or quiet?
    f["lag1_casual_share"] = (raw["casual"] / raw["cnt"]).shift(1)
    for c in ["weathersit", "temp", "hum", "windspeed"]:
        f[f"{c}_lag1"] = raw[c].shift(1)            # today's weather, known this evening
    for c in ["weekday", "workingday", "holiday"]:
        f[c] = raw[c]                               # tomorrow's calendar, known in advance
    f["wd_change"] = raw["workingday"] - raw["workingday"].shift(1)
    doy = raw["dteday"].dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f.iloc[WARMUP_DAYS:].dropna().reset_index(drop=True)


def split(f: pd.DataFrame):
    train = f[f["date"] <= TRAIN_END]
    val = f[(f["date"] >= VAL_START) & (f["date"] <= VAL_END)]
    test = f[(f["date"] >= TEST_START) & (f["date"] <= TEST_END)]
    return train, val, test


# ----------------------------------------------------------------------------- model wrapper
class DeviationFromWeekRegressor(BaseEstimator, RegressorMixin):
    """Learn tomorrow minus the trailing 7-day mean; add the mean back when predicting.

    Rentals grew ~65% from 2011 to 2012. Predicting the deviation keeps the target on a stable
    scale, so both approaches are compared on equal terms.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - X["roll7"].to_numpy())
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + X["roll7"].to_numpy()


# ----------------------------------------------------------------------------- metrics
def forecast_metrics(y, pred, naive, prefix: str) -> dict:
    y, pred, naive = (np.asarray(a, dtype=float) for a in (y, pred, naive))
    err = pred - y
    mae, naive_mae = float(np.mean(np.abs(err))), float(np.mean(np.abs(naive - y)))
    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))),
        f"{prefix}_mape": float(np.mean(np.abs(err) / y) * 100),
        f"{prefix}_naive_mae": naive_mae,
        # > 0 means better than "tomorrow = today"; 0.05 = 5% lower MAE than the benchmark
        f"{prefix}_skill_vs_naive": 1 - mae / naive_mae,
        # under-forecasts leave operations short-staffed, so they are tracked separately
        f"{prefix}_under_forecast_share": float((err < 0).mean()),
        f"{prefix}_bias": float(err.mean()),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-checkout"


# ----------------------------------------------------------------------------- one logged run
def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def run_experiment(approach: str, analyst: str, model, params: dict, run_name: str) -> dict:
    """Fit on train, score on validation, and log everything needed to trace the result."""
    feats = load_features()
    train, val, _ = split(feats)

    with mlflow.start_run(run_name=run_name) as run:
        # who / which approach / which data / which code
        mlflow.set_tags({
            "approach": approach,
            "analyst": analyst,
            "data_sha256": file_sha256(),
            "git_commit": git_commit(),
        })
        # data period, features and settings
        mlflow.log_params({
            "train_start": str(train["date"].min().date()), "train_end": str(train["date"].max().date()),
            "val_start": str(val["date"].min().date()), "val_end": str(val["date"].max().date()),
            "n_train_rows": len(train), "n_val_rows": len(val),
            "features": ",".join(FEATURES), "n_features": len(FEATURES),
            "target": "cnt minus trailing 7-day mean",
            **params,
        })
        # the datasets themselves (shown under "Datasets used" in the MLflow UI)
        mlflow.log_input(mlflow.data.from_pandas(train, source=str(DATA_PATH), targets="y",
                                                 name="train"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(val, source=str(DATA_PATH), targets="y",
                                                 name="validation"), context="validation")

        model.fit(train[FEATURES], train["y"])
        pred = model.predict(val[FEATURES])
        metrics = forecast_metrics(val["y"], pred, val["lag1"], "val")
        metrics["train_mae"] = forecast_metrics(train["y"], model.predict(train[FEATURES]),
                                                train["lag1"], "train")["train_mae"]
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"date": val["date"].dt.date, "actual": val["y"], "forecast": pred.round(0),
                          "naive_forecast": val["lag1"]}).to_csv(f"{tmp}/val_predictions.csv", index=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(val["date"], val["y"], label="actual", color="#222")
            ax.plot(val["date"], val["lag1"], label="naive: tomorrow = today", color="#aaa", lw=1)
            ax.plot(val["date"], pred, label=approach, color="#1f6feb")
            ax.set_title(f"{run_name}: val MAE {metrics['val_mae']:.0f} "
                         f"vs naive {metrics['val_naive_mae']:.0f}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{tmp}/val_forecast.png", dpi=110)
            plt.close(fig)
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        logged = mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(val[FEATURES], pred),
            input_example=val[FEATURES].head(3),
            code_paths=[str(ROOT / "common.py")],        # the wrapper class travels with the model
            # MLflow 3 saves sklearn models with skops, which loads only allow-listed types.
            # We trained this object ourselves just now, so we allow exactly its types.
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)) or None,
            pip_requirements=["scikit-learn==1.9.0", "skops==0.16.0", "pandas==3.0.5", "numpy==2.5.3"],
        )
        mlflow.set_tag("model_uri", logged.model_uri)

    print(f"  {run_name:42s} val MAE {metrics['val_mae']:5.0f} | naive {metrics['val_naive_mae']:4.0f} "
          f"| skill {metrics['val_skill_vs_naive']:+.3f} | run {run.info.run_id}")
    return metrics
