"""Approach A (analyst A): Ridge regression.

A linear model on lagged rentals, yesterday's weather and tomorrow's calendar. Linear models
can extrapolate, which matters because 2012 volumes exceed anything seen in 2011.
One MLflow run per value of the regularisation strength alpha.

    python train_ridge.py
    python train_ridge.py --alphas 0.3 1 3
"""
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATEGORICAL, NUMERIC, DeviationFromWeekRegressor, run_experiment, setup_mlflow


def build(alpha: float):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return DeviationFromWeekRegressor(Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", nargs="+", type=float, default=[1.0, 3.0, 10.0, 30.0])
    p.add_argument("--analyst", default="analyst_a")
    a = p.parse_args()

    setup_mlflow()
    print("Approach A - Ridge regression")
    for alpha in a.alphas:
        run_experiment(
            approach="ridge_regression", analyst=a.analyst, model=build(alpha),
            params={"model_type": "Ridge", "alpha": alpha, "scaling": "StandardScaler",
                    "categorical_encoding": "one-hot"},
            run_name=f"ridge-alpha{alpha:g}",
        )


if __name__ == "__main__":
    main()
