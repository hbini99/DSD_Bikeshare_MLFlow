"""Verify the champion the way operations would use it: retrieve it by name, trace it back
to its training run, and check its forecasts against actuals and the naive benchmark.

    python verify_champion.py
"""
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from common import FEATURES, MODEL_NAME, forecast_metrics, load_features, setup_mlflow, split


def main():
    setup_mlflow()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)
    p = run.data.params

    print(f"{MODEL_NAME}@champion -> version {mv.version}")
    print(f"  produced by run   {mv.run_id} ({run.info.run_name})")
    print(f"  approach/analyst  {run.data.tags['approach']} / {run.data.tags['analyst']}")
    print(f"  training data     {p['train_start']} to {p['train_end']}  (sha256 "
          f"{run.data.tags['data_sha256'][:12]}...)")
    settings = {k: v for k, v in p.items() if not k.startswith(("train_", "val_", "n_", "features"))}
    print("  settings          " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    print(f"  benchmark check   {mv.tags.get('benchmark_check')} "
          f"(test MAE {mv.tags.get('test_mae')} vs naive {mv.tags.get('test_naive_mae')})")

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    _, _, test = split(load_features())
    pred = model.predict(test[FEATURES])
    m = forecast_metrics(test["y"], pred, test["lag1"], "test")
    print(f"\nRe-scored test period: MAE {m['test_mae']:.0f} vs naive {m['test_naive_mae']:.0f}, "
          f"under-forecast on {m['test_under_forecast_share']:.0%} of days")
    assert f"{m['test_mae']:.0f}" == mv.tags.get("test_mae"), "result does not reproduce!"
    print("Reproduces the MAE recorded at registration.")

    last = pd.DataFrame({"date": test["date"].dt.date, "actual": test["y"].astype(int),
                         "champion": pred.round(0).astype(int), "naive": test["lag1"].astype(int)}).tail(7)
    print("\nLast 7 days of 2012:")
    print(last.to_string(index=False))


if __name__ == "__main__":
    main()
