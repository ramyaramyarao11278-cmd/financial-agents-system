from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from agents.data_engineer import DataEngineerAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from utils.evaluation import find_best_threshold, regression_metrics


@dataclass
class ExperimentDataset:
    symbol: str
    feature_names: List[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_return_train: np.ndarray
    y_return_val: np.ndarray
    y_return_test: np.ndarray
    y_direction_train: np.ndarray
    y_direction_val: np.ndarray
    y_direction_test: np.ndarray
    test_dates: List[pd.Timestamp]
    test_anchor_close: np.ndarray
    past_horizon_return_test: np.ndarray


def build_experiment_dataset(symbol: str = "CSI300", feature_subset: Optional[List[str]] = None) -> ExperimentDataset:
    data_result = DataEngineerAgent().run({})
    sentiment_result = SentimentAnalystAgent().run(
        {
            "sentiment_data_by_symbol": data_result["result"]["sentiment_data_by_symbol"],
            "news": data_result["result"].get("news", []),
        }
    )

    technical_agent = TechnicalAnalystAgent()
    price_records = data_result["result"]["data_by_symbol"][symbol]
    df = pd.DataFrame(price_records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if "Symbol" in df.columns:
        df = df.drop(columns=["Symbol"])

    indicator_results = technical_agent._calculate_indicators(
        df,
        ["sma", "ema", "rsi", "macd", "bollinger", "stochastics"],
    )
    enhanced_df = technical_agent._create_enhanced_dataframe(df, indicator_results)
    symbol_sentiment = sentiment_result["result"]["by_symbol"][symbol]
    sentiment_df = technical_agent._prepare_sentiment_data(symbol_sentiment, enhanced_df)
    combined_df = technical_agent._build_combined_dataframe(enhanced_df, sentiment_df)
    target_df = technical_agent._build_prediction_targets(combined_df, symbol)
    X, y_return, y_direction, sample_dates, anchor_close, feature_names = technical_agent._create_window_dataset(
        target_df,
        sequence_length=20,
    )

    if feature_subset is not None:
        feature_index = [feature_names.index(name) for name in feature_subset]
        X = X[:, :, feature_index]
        feature_names = [feature_names[idx] for idx in feature_index]

    horizon = technical_agent._get_target_horizon(symbol)
    daily_return = target_df["Close"].pct_change()
    past_horizon_return = sum(daily_return.shift(step) for step in range(0, horizon))
    past_horizon_return = past_horizon_return.loc[target_df.index].to_numpy(dtype=np.float32)[19:]

    split_idx = max(1, min(len(X) - 1, int(len(X) * 0.8)))
    X_train_all, X_test = X[:split_idx], X[split_idx:]
    y_return_train_all, y_return_test = y_return[:split_idx], y_return[split_idx:]
    y_direction_train_all, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
    test_dates = sample_dates[split_idx:]
    test_anchor_close = anchor_close[split_idx:]
    past_horizon_return_test = past_horizon_return[split_idx:]

    val_size = max(1, int(len(X_train_all) * 0.2))
    train_end = len(X_train_all) - val_size

    return ExperimentDataset(
        symbol=symbol,
        feature_names=feature_names,
        X_train=X_train_all[:train_end],
        X_val=X_train_all[train_end:],
        X_test=X_test,
        y_return_train=y_return_train_all[:train_end],
        y_return_val=y_return_train_all[train_end:],
        y_return_test=y_return_test,
        y_direction_train=y_direction_train_all[:train_end],
        y_direction_val=y_direction_train_all[train_end:],
        y_direction_test=y_direction_test,
        test_dates=test_dates,
        test_anchor_close=test_anchor_close,
        past_horizon_return_test=past_horizon_return_test,
    )


def classification_metrics_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_val_true: Optional[np.ndarray] = None,
    y_val_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    threshold = 0.5
    if y_val_true is not None and y_val_prob is not None:
        threshold, _, _ = find_best_threshold(y_val_true, y_val_prob)

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Threshold": float(threshold),
    }
    return metrics


def combine_metrics(
    regression_true: np.ndarray,
    regression_pred: np.ndarray,
    classification_true: np.ndarray,
    classification_prob: np.ndarray,
    val_true: Optional[np.ndarray] = None,
    val_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    reg = regression_metrics(regression_true, regression_pred)
    cls = classification_metrics_from_probabilities(classification_true, classification_prob, val_true, val_prob)
    return {
        "R2": reg["R2"],
        "RMSE": reg["RMSE"],
        "MAE": reg["MAE"],
        "AUC": cls["AUC"],
        "Balanced_Acc": cls["Balanced_Acc"],
        "F1": cls["F1"],
    }
