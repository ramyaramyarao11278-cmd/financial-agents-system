from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from utils.evaluation import find_best_threshold, regression_metrics
from utils.experiment_utils import build_experiment_dataset, classification_metrics_from_probabilities
from utils.transformer_model import create_transformer_model

try:
    from pmdarima import auto_arima  # type: ignore
except Exception:
    auto_arima = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None


def fit_arima_forecaster(history: List[float]):
    if auto_arima is not None:
        return auto_arima(
            history,
            seasonal=False,
            max_p=3,
            max_q=3,
            stepwise=True,
            suppress_warnings=True,
        )

    if ARIMA is None:
        raise RuntimeError("Ensemble regression requires either pmdarima or statsmodels.")

    best_result = None
    best_aic = np.inf
    for order in [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (1, 1, 0)]:
        try:
            result = ARIMA(history, order=order).fit()
            if float(result.aic) < best_aic:
                best_aic = float(result.aic)
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("Unable to fit fallback ARIMA model with statsmodels.")
    return best_result


def forecast_arima(model, n_periods: int) -> np.ndarray:
    if auto_arima is not None:
        return np.asarray(model.predict(n_periods=n_periods), dtype=float)
    return np.asarray(model.forecast(steps=n_periods), dtype=float)


def rolling_auto_arima_predict(
    history_values: np.ndarray,
    eval_values: np.ndarray,
    refit_interval: int = 50,
) -> np.ndarray:
    history = list(np.asarray(history_values, dtype=float))
    eval_values = np.asarray(eval_values, dtype=float)
    predictions: List[float] = []

    for start in range(0, len(eval_values), refit_interval):
        stop = min(start + refit_interval, len(eval_values))
        model = fit_arima_forecaster(history)
        block_predictions = forecast_arima(model, stop - start)
        predictions.extend([float(value) for value in block_predictions])
        for idx in range(start, stop):
            history.append(float(eval_values[idx]))

    return np.asarray(predictions, dtype=float)


def train_transformer_regressor(dataset) -> Tuple[object, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    reg_model = create_transformer_model(sequence_length=20, num_features=dataset.X_train.shape[-1])
    reg_model.train(
        dataset.X_train,
        dataset.y_return_train,
        X_val=dataset.X_val,
        y_val=dataset.y_return_val,
        epochs=500,
        batch_size=min(32, len(dataset.X_train)),
    )

    train_pred = reg_model.predict(dataset.X_train)
    val_pred = reg_model.predict(dataset.X_val)
    test_pred = reg_model.predict(dataset.X_test)

    train_metrics = regression_metrics(dataset.y_return_train, train_pred)
    val_metrics = regression_metrics(dataset.y_return_val, val_pred)
    test_metrics = regression_metrics(dataset.y_return_test, test_pred)
    pred_stats = {
        "mean": float(np.mean(test_pred)),
        "std": float(np.std(test_pred)),
        "min": float(np.min(test_pred)),
        "max": float(np.max(test_pred)),
    }

    print(
        "[Transformer回归诊断] "
        f"params={reg_model.model.count_params()} "
        f"train_R2={train_metrics['R2']:.6f} "
        f"val_R2={val_metrics['R2']:.6f} "
        f"test_R2={test_metrics['R2']:.6f} "
        f"pred_stats={pred_stats}"
    )
    return reg_model, train_metrics, val_metrics, test_metrics, pred_stats


def build_classifier_features(dataset, reg_model) -> Dict[str, np.ndarray]:
    train_hidden = reg_model.extract_encoder_features(dataset.X_train)
    val_hidden = reg_model.extract_encoder_features(dataset.X_val)
    test_hidden = reg_model.extract_encoder_features(dataset.X_test)

    train_last = dataset.X_train[:, -1, :]
    val_last = dataset.X_val[:, -1, :]
    test_last = dataset.X_test[:, -1, :]

    return {
        "train_hidden": train_hidden,
        "val_hidden": val_hidden,
        "test_hidden": test_hidden,
        "train_last": train_last,
        "val_last": val_last,
        "test_last": test_last,
        "train_fused": np.concatenate([train_hidden, train_last], axis=1),
        "val_fused": np.concatenate([val_hidden, val_last], axis=1),
        "test_fused": np.concatenate([test_hidden, test_last], axis=1),
    }


def evaluate_classification_candidate(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> Dict[str, float]:
    threshold, _, _ = find_best_threshold(y_val, val_prob)
    metrics = classification_metrics_from_probabilities(y_test, test_prob, y_val, val_prob)
    metrics["Threshold"] = float(threshold)
    return metrics


def run_classification_ensemble(dataset, reg_model) -> Dict[str, Dict[str, float]]:
    features = build_classifier_features(dataset, reg_model)

    pure_lr = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", random_state=42)
    pure_lr.fit(features["train_last"], dataset.y_direction_train)
    pure_lr_val = pure_lr.predict_proba(features["val_last"])[:, 1]
    pure_lr_test = pure_lr.predict_proba(features["test_last"])[:, 1]
    pure_lr_metrics = evaluate_classification_candidate(
        dataset.y_direction_val,
        pure_lr_val,
        dataset.y_direction_test,
        pure_lr_test,
    )

    fused_lr = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", random_state=42)
    fused_lr.fit(features["train_fused"], dataset.y_direction_train)
    fused_lr_val = fused_lr.predict_proba(features["val_fused"])[:, 1]
    fused_lr_test = fused_lr.predict_proba(features["test_fused"])[:, 1]
    fused_lr_metrics = evaluate_classification_candidate(
        dataset.y_direction_val,
        fused_lr_val,
        dataset.y_direction_test,
        fused_lr_test,
    )

    gbt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    gbt.fit(features["train_fused"], dataset.y_direction_train)
    gbt_val = gbt.predict_proba(features["val_fused"])[:, 1]
    gbt_test = gbt.predict_proba(features["test_fused"])[:, 1]
    gbt_metrics = evaluate_classification_candidate(
        dataset.y_direction_val,
        gbt_val,
        dataset.y_direction_test,
        gbt_test,
    )

    ensemble_val = 0.5 * fused_lr_val + 0.5 * gbt_val
    ensemble_test = 0.5 * fused_lr_test + 0.5 * gbt_test
    ensemble_metrics = evaluate_classification_candidate(
        dataset.y_direction_val,
        ensemble_val,
        dataset.y_direction_test,
        ensemble_test,
    )

    print("\n[融合分类结果]")
    print(
        f"纯LR(19维): AUC={pure_lr_metrics['AUC']:.6f}, "
        f"Balanced_Acc={pure_lr_metrics['Balanced_Acc']:.6f}, F1={pure_lr_metrics['F1']:.6f}"
    )
    print(
        f"LR+Transformer特征: AUC={fused_lr_metrics['AUC']:.6f}, "
        f"Balanced_Acc={fused_lr_metrics['Balanced_Acc']:.6f}, F1={fused_lr_metrics['F1']:.6f}"
    )
    print(
        f"GBT+Transformer特征: AUC={gbt_metrics['AUC']:.6f}, "
        f"Balanced_Acc={gbt_metrics['Balanced_Acc']:.6f}, F1={gbt_metrics['F1']:.6f}"
    )
    print(
        f"融合(0.5LR+0.5GBT): AUC={ensemble_metrics['AUC']:.6f}, "
        f"Balanced_Acc={ensemble_metrics['Balanced_Acc']:.6f}, F1={ensemble_metrics['F1']:.6f}"
    )

    return {
        "pure_lr": pure_lr_metrics,
        "fused_lr": fused_lr_metrics,
        "gbt": gbt_metrics,
        "ensemble": ensemble_metrics,
    }


def run_regression_ensemble(dataset, transformer_model) -> Dict[str, float]:
    arima_val = rolling_auto_arima_predict(dataset.y_return_train, dataset.y_return_val)
    arima_test = rolling_auto_arima_predict(
        np.concatenate([dataset.y_return_train, dataset.y_return_val]),
        dataset.y_return_test,
    )
    transformer_val = transformer_model.predict(dataset.X_val)
    transformer_test = transformer_model.predict(dataset.X_test)

    best_alpha = 0.0
    best_metrics = None
    best_score = -np.inf

    for alpha in np.arange(0.0, 1.0001, 0.05):
        val_pred = alpha * arima_val + (1.0 - alpha) * transformer_val
        val_metrics = regression_metrics(dataset.y_return_val, val_pred)
        if val_metrics["R2"] > best_score:
            best_score = val_metrics["R2"]
            best_alpha = float(alpha)
            best_metrics = val_metrics

    test_pred = best_alpha * arima_test + (1.0 - best_alpha) * transformer_test
    test_metrics = regression_metrics(dataset.y_return_test, test_pred)
    print(
        "\n[融合回归结果] "
        f"best_alpha={best_alpha:.2f} "
        f"val_R2={best_metrics['R2']:.6f} "
        f"test_R2={test_metrics['R2']:.6f} "
        f"RMSE={test_metrics['RMSE']:.6f} "
        f"MAE={test_metrics['MAE']:.6f}"
    )
    return {"alpha": best_alpha, **test_metrics}


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def run_ablation_ensemble_auc(config: str, feature_subset: Optional[List[str]] = None) -> float:
    dataset = build_experiment_dataset(symbol="CSI300", feature_subset=feature_subset)
    reg_model, _, _, _, _ = train_transformer_regressor(dataset)
    cls_results = run_classification_ensemble(dataset, reg_model)
    print(f"[Ablation融合] {config} ensemble_auc={cls_results['ensemble']['AUC']:.6f}")
    return float(cls_results["ensemble"]["AUC"])


def build_feature_subset(mode: str, full_feature_names: List[str]) -> Optional[List[str]]:
    if mode == "price_only":
        return [name for name in full_feature_names if not name.startswith("news_") and not name.startswith("sentiment_")]
    if mode == "sentiment_only":
        selected = [name for name in full_feature_names if name.startswith("news_") or name.startswith("sentiment_")]
        selected += [name for name in ["Close", "lag_return_1", "lag_return_2", "lag_return_3"] if name in full_feature_names]
        return list(dict.fromkeys(selected))
    if mode == "no_lag_return":
        return [name for name in full_feature_names if name not in {"lag_return_1", "lag_return_2", "lag_return_3"}]
    return None


def create_final_tables(
    transformer_reg_metrics: Dict[str, float],
    transformer_cls_metrics: Dict[str, float],
    ensemble_reg_metrics: Dict[str, float],
    ensemble_cls_metrics: Dict[str, float],
    ablation_ensemble_auc: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_v3 = pd.read_csv(os.path.join("outputs", "baseline_comparison_v3.csv"))
    ablation_v3 = pd.read_csv(os.path.join("outputs", "ablation_results_v3.csv"))

    baseline_lookup = baseline_v3.set_index("Model")
    final_rows = []
    for model_name in ["Naive", "LR", "ARIMA", "LSTM", "GRU", "RF"]:
        row = baseline_lookup.loc[model_name]
        final_rows.append(
            {
                "Model": model_name,
                "R2": format_metric(float(row["R2"])),
                "RMSE": format_metric(float(row["RMSE"])),
                "MAE": format_metric(float(row["MAE"])),
                "AUC": format_metric(float(row["AUC"])),
                "Balanced_Acc": format_metric(float(row["Balanced_Acc"])),
                "F1": format_metric(float(row["F1"])),
            }
        )

    final_rows.append(
        {
            "Model": "Transformer_Regression",
            "R2": format_metric(transformer_reg_metrics["R2"]),
            "RMSE": format_metric(transformer_reg_metrics["RMSE"]),
            "MAE": format_metric(transformer_reg_metrics["MAE"]),
            "AUC": "-",
            "Balanced_Acc": "-",
            "F1": "-",
        }
    )
    final_rows.append(
        {
            "Model": "Transformer_Classification",
            "R2": "-",
            "RMSE": "-",
            "MAE": "-",
            "AUC": format_metric(transformer_cls_metrics["AUC"]),
            "Balanced_Acc": format_metric(transformer_cls_metrics["Balanced_Acc"]),
            "F1": format_metric(transformer_cls_metrics["F1"]),
        }
    )
    final_rows.append(
        {
            "Model": "Ensemble_Regression",
            "R2": format_metric(ensemble_reg_metrics["R2"]),
            "RMSE": format_metric(ensemble_reg_metrics["RMSE"]),
            "MAE": format_metric(ensemble_reg_metrics["MAE"]),
            "AUC": "-",
            "Balanced_Acc": "-",
            "F1": "-",
        }
    )
    final_rows.append(
        {
            "Model": "Ensemble_Classification",
            "R2": "-",
            "RMSE": "-",
            "MAE": "-",
            "AUC": format_metric(ensemble_cls_metrics["AUC"]),
            "Balanced_Acc": format_metric(ensemble_cls_metrics["Balanced_Acc"]),
            "F1": format_metric(ensemble_cls_metrics["F1"]),
        }
    )
    final_baseline = pd.DataFrame(final_rows)

    ablation_df = ablation_v3.copy()
    ablation_df["Ensemble_AUC"] = ablation_df["Config"].map(ablation_ensemble_auc)
    final_ablation = ablation_df[["Config", "R2", "RMSE", "MAE", "AUC", "Balanced_Acc", "F1", "Feature_Count", "Ensemble_AUC"]]

    os.makedirs("outputs", exist_ok=True)
    final_baseline.to_csv(os.path.join("outputs", "final_baseline_comparison.csv"), index=False, encoding="utf-8-sig")
    final_ablation.to_csv(os.path.join("outputs", "final_ablation.csv"), index=False, encoding="utf-8-sig")

    print("\n[最终基线表]")
    print(final_baseline.to_string(index=False))
    print("\n[最终消融表]")
    print(final_ablation.to_string(index=False))

    return final_baseline, final_ablation


def main() -> None:
    dataset = build_experiment_dataset(symbol="CSI300")
    reg_model, _, _, transformer_reg_metrics, reg_stats = train_transformer_regressor(dataset)
    classification_results = run_classification_ensemble(dataset, reg_model)
    ensemble_reg_metrics = run_regression_ensemble(dataset, reg_model)

    full_feature_names = dataset.feature_names
    ablation_ensemble_auc = {}
    for config in ["price_only", "sentiment_only", "no_lag_return", "full_model"]:
        subset = build_feature_subset(config, full_feature_names)
        ablation_ensemble_auc[config] = run_ablation_ensemble_auc(config, subset)

    transformer_cls_row = pd.read_csv(os.path.join("outputs", "baseline_comparison_v3.csv")).set_index("Model").loc["Transformer"]
    transformer_cls_metrics = {
        "AUC": float(transformer_cls_row["AUC"]),
        "Balanced_Acc": float(transformer_cls_row["Balanced_Acc"]),
        "F1": float(transformer_cls_row["F1"]),
    }

    create_final_tables(
        transformer_reg_metrics=transformer_reg_metrics,
        transformer_cls_metrics=transformer_cls_metrics,
        ensemble_reg_metrics=ensemble_reg_metrics,
        ensemble_cls_metrics=classification_results["ensemble"],
        ablation_ensemble_auc=ablation_ensemble_auc,
    )

    print(
        "\n[中文总结] "
        f"分类融合相对纯LR的AUC增量={classification_results['ensemble']['AUC'] - classification_results['pure_lr']['AUC']:.4f}；"
        f"回归融合相对纯ARIMA的R2增量={ensemble_reg_metrics['R2'] - pd.read_csv(os.path.join('outputs', 'baseline_comparison_v3.csv')).set_index('Model').loc['ARIMA', 'R2']:.4f}；"
        f"Transformer回归预测std={reg_stats['std']:.4f}"
    )


if __name__ == "__main__":
    main()
