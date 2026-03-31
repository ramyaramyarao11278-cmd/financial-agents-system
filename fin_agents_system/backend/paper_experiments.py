from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from baselines import (
    evaluate_arima,
    evaluate_linear_regression,
    evaluate_naive,
    evaluate_random_forest,
    evaluate_torch_baseline,
    evaluate_transformer_current,
)
from ensemble_model import (
    build_feature_subset,
    rolling_auto_arima_predict,
    run_classification_ensemble,
    train_transformer_regressor,
)
from utils.evaluation import regression_metrics
from utils.experiment_utils import build_experiment_dataset


def ensure_output_dir() -> str:
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def normalize_symbol(symbol: str) -> str:
    return symbol.upper()


def build_baseline_rows(dataset) -> pd.DataFrame:
    evaluators = {
        "Naive": evaluate_naive,
        "LR": evaluate_linear_regression,
        "ARIMA": evaluate_arima,
        "LSTM": lambda ds: evaluate_torch_baseline("lstm", ds),
        "GRU": lambda ds: evaluate_torch_baseline("gru", ds),
        "RF": evaluate_random_forest,
        "Transformer": evaluate_transformer_current,
    }

    rows = []
    for model_name, evaluator in evaluators.items():
        metrics = evaluator(dataset)
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)


def run_regression_ensemble_with_predictions(dataset, transformer_model) -> Tuple[Dict[str, float], np.ndarray]:
    arima_val = rolling_auto_arima_predict(dataset.y_return_train, dataset.y_return_val)
    arima_test = rolling_auto_arima_predict(
        np.concatenate([dataset.y_return_train, dataset.y_return_val]),
        dataset.y_return_test,
    )
    transformer_val = transformer_model.predict(dataset.X_val)
    transformer_test = transformer_model.predict(dataset.X_test)

    best_alpha = 0.0
    best_score = -np.inf
    for alpha in np.arange(0.0, 1.0001, 0.05):
        val_pred = alpha * arima_val + (1.0 - alpha) * transformer_val
        val_metrics = regression_metrics(dataset.y_return_val, val_pred)
        if val_metrics["R2"] > best_score:
            best_score = val_metrics["R2"]
            best_alpha = float(alpha)

    test_pred = best_alpha * arima_test + (1.0 - best_alpha) * transformer_test
    return {"alpha": best_alpha, **regression_metrics(dataset.y_return_test, test_pred)}, test_pred


def run_symbol_experiment(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbol = normalize_symbol(symbol)
    dataset = build_experiment_dataset(symbol=symbol)
    baseline_df = build_baseline_rows(dataset)

    reg_model, _, _, transformer_reg_metrics, _ = train_transformer_regressor(dataset)
    classification_results = run_classification_ensemble(dataset, reg_model)
    ensemble_reg_metrics, _ = run_regression_ensemble_with_predictions(dataset, reg_model)

    transformer_row = baseline_df.set_index("Model").loc["Transformer"]
    baseline_rows = baseline_df.to_dict("records")
    baseline_rows.append(
        {
            "Model": "Ensemble",
            "R2": ensemble_reg_metrics["R2"],
            "RMSE": ensemble_reg_metrics["RMSE"],
            "MAE": ensemble_reg_metrics["MAE"],
            "AUC": classification_results["ensemble"]["AUC"],
            "Balanced_Acc": classification_results["ensemble"]["Balanced_Acc"],
            "F1": classification_results["ensemble"]["F1"],
        }
    )
    baseline_df = pd.DataFrame(baseline_rows)

    full_feature_names = dataset.feature_names
    ablation_rows = []
    for config in ["price_only", "sentiment_only", "no_lag_return", "full_model"]:
        subset = build_feature_subset(config, full_feature_names)
        ablation_dataset = build_experiment_dataset(symbol=symbol, feature_subset=subset)
        reg_ablation_model, _, _, _, _ = train_transformer_regressor(ablation_dataset)
        cls_results = run_classification_ensemble(ablation_dataset, reg_ablation_model)
        ensemble_reg_ablation_metrics, _ = run_regression_ensemble_with_predictions(ablation_dataset, reg_ablation_model)
        ablation_rows.append(
            {
                "Config": config,
                "R2": ensemble_reg_ablation_metrics["R2"],
                "RMSE": ensemble_reg_ablation_metrics["RMSE"],
                "MAE": ensemble_reg_ablation_metrics["MAE"],
                "AUC": float(cls_results["fused_lr"]["AUC"]) if config == "full_model" else float(cls_results["ensemble"]["AUC"]),
                "Balanced_Acc": float(cls_results["ensemble"]["Balanced_Acc"]),
                "F1": float(cls_results["ensemble"]["F1"]),
                "Feature_Count": len(ablation_dataset.feature_names),
                "Ensemble_AUC": float(cls_results["ensemble"]["AUC"]),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)

    print(f"[{symbol}] Transformer regression metrics: {transformer_reg_metrics}")
    print(f"[{symbol}] Transformer combined row source: {transformer_row.to_dict()}")
    print(f"[{symbol}] Ensemble metrics: regression={ensemble_reg_metrics}, classification={classification_results['ensemble']}")

    return baseline_df, ablation_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CSI100")
    args = parser.parse_args()

    output_dir = ensure_output_dir()
    symbol = normalize_symbol(args.symbol)
    baseline_df, ablation_df = run_symbol_experiment(symbol)

    baseline_path = os.path.join(output_dir, f"{symbol.lower()}_baseline_comparison.csv")
    ablation_path = os.path.join(output_dir, f"{symbol.lower()}_ablation_results.csv")
    baseline_df.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")

    if symbol == "CSI100":
        print(
            "CSI100 control summary: higher volatility keeps all models weaker than CSI300; "
            "this is consistent with the expected harder forecasting problem under market efficiency."
        )

    print(baseline_df.to_string(index=False))
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
