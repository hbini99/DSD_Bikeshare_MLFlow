"""Pick the champion and register it in the MLflow Model Registry.

1. Rank every training run in the experiment by validation MAE (both approaches).
2. Register the best run as a new version of `bikeshare-next-day-rentals`.
   The version keeps a link to the run that produced it (data period, features, settings).
3. Benchmark check: load that version BY NAME and score the untouched test period
   (Oct-Dec 2012) against "tomorrow = today". Only if it wins does it get the
   @champion alias that operations uses. Otherwise it is tagged as rejected.

    python register_champion.py
"""
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from common import (EXPERIMENT, FEATURES, MODEL_NAME, TEST_END, TEST_START, forecast_metrics,
                    load_features, setup_mlflow, split)


def main():
    setup_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], filter_string="tags.approach != ''",
                              order_by=["metrics.val_mae ASC"])
    runs = [r for r in runs if "val_mae" in r.data.metrics]
    if not runs:
        raise SystemExit("No training runs found - run train_ridge.py and train_gbm.py first.")

    print("Validation leaderboard (Jul-Sep 2012):")
    for r in runs:
        m = r.data.metrics
        print(f"  {r.data.tags['approach']:18s} {r.info.run_name:32s} MAE {m['val_mae']:5.0f} "
              f"(naive {m['val_naive_mae']:.0f}, skill {m['val_skill_vs_naive']:+.3f})")

    best = runs[0]
    existing = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                if v.run_id == best.info.run_id]
    if existing:                          # re-running the script must not create duplicates
        mv = existing[0]
        print(f"\nBest run is already registered as version {mv.version} - re-checking it.")
    else:
        model_uri = best.data.tags.get("model_uri", f"runs:/{best.info.run_id}/model")
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"\nRegistered {MODEL_NAME} version {mv.version} from run {best.info.run_id} "
              f"({best.info.run_name})")
    client.update_registered_model(
        MODEL_NAME, "Next-day total rentals for evening staffing and fleet planning (Capital Bikeshare).")
    for k, v in {"source_run_id": best.info.run_id, "approach": best.data.tags["approach"],
                 "analyst": best.data.tags["analyst"], "data_sha256": best.data.tags["data_sha256"],
                 "val_mae": f"{best.data.metrics['val_mae']:.0f}"}.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, v)

    # ---- benchmark check on the held-out test period -----------------------------------------
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    _, _, test = split(load_features())
    m = forecast_metrics(test["y"], model.predict(test[FEATURES]), test["lag1"], "test")
    passed = m["test_mae"] < m["test_naive_mae"]

    with mlflow.start_run(run_name=f"champion-check-v{mv.version}"):
        mlflow.set_tags({"stage": "champion_check", "model_name": MODEL_NAME,
                         "model_version": mv.version, "source_run_id": best.info.run_id,
                         "result": "passed" if passed else "failed"})
        mlflow.log_params({"test_start": TEST_START, "test_end": TEST_END, "n_test_rows": len(test),
                           "benchmark": "naive: tomorrow = today"})
        mlflow.log_metrics(m)

    client.set_model_version_tag(MODEL_NAME, mv.version, "test_mae", f"{m['test_mae']:.0f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_naive_mae", f"{m['test_naive_mae']:.0f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "benchmark_check",
                                 "passed" if passed else "failed")
    print(f"Benchmark check ({TEST_START} to {TEST_END}, {len(test)} days): "
          f"MAE {m['test_mae']:.0f} vs naive {m['test_naive_mae']:.0f} "
          f"(skill {m['test_skill_vs_naive']:+.3f})")

    if passed:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        client.update_model_version(
            MODEL_NAME, mv.version,
            f"Champion: beats the naive benchmark on {TEST_START}..{TEST_END} "
            f"(MAE {m['test_mae']:.0f} vs {m['test_naive_mae']:.0f}).")
        print(f"PASSED -> {MODEL_NAME}@champion = version {mv.version}")
    else:
        print("FAILED -> version kept but NOT made champion; the previous champion (if any) stays.")


if __name__ == "__main__":
    main()
