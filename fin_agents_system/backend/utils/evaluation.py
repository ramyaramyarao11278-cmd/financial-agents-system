import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, y_true))
    ) * 100

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


def trend_labels(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int)
    return (values > 0).astype(int)


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        metrics["AUC_ROC"] = float(auc(fpr, tpr))

    return metrics


def classification_metrics_from_scores(y_true, y_score, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = classification_metrics(y_true, y_pred)
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["AUC_ROC"] = float(auc(fpr, tpr))
    metrics["Threshold"] = float(threshold)
    return metrics


def find_best_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        return 0.5, {}, []

    thresholds = np.arange(0.30, 0.7001, 0.005)
    best_threshold = 0.5
    best_metrics = {}
    best_score = -1.0
    ranked_candidates = []

    for threshold in thresholds:
        metrics = classification_metrics_from_scores(y_true, y_score, threshold=threshold)
        composite_score = (
            0.6 * metrics.get("Balanced_Accuracy", 0.0) +
            0.4 * metrics.get("F1", 0.0)
        )
        ranked_candidates.append(
            {
                "Threshold": float(threshold),
                "Balanced_Accuracy": float(metrics.get("Balanced_Accuracy", 0.0)),
                "F1": float(metrics.get("F1", 0.0)),
                "Accuracy": float(metrics.get("Accuracy", 0.0)),
                "Score": float(composite_score),
            }
        )
        if composite_score > best_score:
            best_score = composite_score
            best_threshold = float(threshold)
            best_metrics = metrics

    ranked_candidates.sort(
        key=lambda item: (item["Score"], item["Balanced_Accuracy"], item["F1"], item["Accuracy"]),
        reverse=True,
    )
    return best_threshold, best_metrics, ranked_candidates[:5]


def confusion_matrix_values(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return matrix


def export_metrics_to_csv(
    actual_prices,
    predicted_prices,
    path="model_metrics.csv",
    symbol=None,
    y_true_cls=None,
    y_pred_cls=None,
    y_score=None,
    threshold=None,
):
    actual_prices = np.asarray(actual_prices, dtype=float)
    predicted_prices = np.asarray(predicted_prices, dtype=float)

    rows = []

    reg = regression_metrics(actual_prices, predicted_prices)
    for k, v in reg.items():
        rows.append({
            "symbol": symbol or "",
            "task": "return_regression",
            "metric": k,
            "value": v
        })

    if y_true_cls is None:
        y_true = trend_labels(actual_prices)
    else:
        y_true = np.asarray(y_true_cls).astype(int)

    if y_pred_cls is None:
        y_pred = trend_labels(predicted_prices)
    else:
        y_pred = np.asarray(y_pred_cls).astype(int)

    cls = classification_metrics(y_true, y_pred)
    if y_score is not None and len(y_true) > 0 and len(np.unique(y_true)) > 1:
        y_score = np.asarray(y_score, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        cls["AUC_ROC"] = float(auc(fpr, tpr))
    if threshold is not None:
        cls["Threshold"] = float(threshold)

    for k, v in cls.items():
        rows.append({
            "symbol": symbol or "",
            "task": "trend_classification",
            "metric": k,
            "value": v
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
