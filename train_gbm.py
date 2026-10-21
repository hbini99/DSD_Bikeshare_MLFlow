"""Approach B (analyst B): gradient-boosted trees (scikit-learn HistGradientBoostingRegressor).

Trees capture non-linear effects and interactions (e.g. rain on a weekend vs a weekday)
without manual feature crosses. One MLflow run per hyperparameter setting.

    python train_gbm.py
"""
import argparse

from sklearn.ensemble import HistGradientBoostingRegressor

from common import CATEGORICAL, NUMERIC, DeviationFromWeekRegressor, run_experiment, setup_mlflow

GRID = [
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 20},
    {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 7, "min_samples_leaf": 30},
    {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 40},
]


def build(hp: dict):
    cat_idx = [len(NUMERIC) + i for i in range(len(CATEGORICAL))]   # weekday, weather as categories
    return DeviationFromWeekRegressor(HistGradientBoostingRegressor(
        **hp, l2_regularization=1.0, categorical_features=cat_idx, random_state=42))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyst", default="analyst_b")
    a = p.parse_args()

    setup_mlflow()
    print("Approach B - Gradient-boosted trees")
    for hp in GRID:
        run_experiment(
            approach="gradient_boosting", analyst=a.analyst, model=build(hp),
            params={"model_type": "HistGradientBoostingRegressor", **hp,
                    "l2_regularization": 1.0, "random_state": 42},
            run_name=f"gbm-lr{hp['learning_rate']}-leaves{hp['max_leaf_nodes']}-it{hp['max_iter']}",
        )


if __name__ == "__main__":
    main()
