from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.data_engineer import DataEngineerAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent


def build_symbol_dataset(symbol: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
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

    next_return = combined_df["Close"].pct_change().shift(-1)

    dataset = combined_df.copy()
    dataset["next_return"] = next_return
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    feature_columns = [col for col in dataset.columns if col != "next_return"]
    X = dataset[feature_columns].astype(float)
    returns = dataset["next_return"].astype(float)
    close_prices = dataset["Close"].astype(float)
    return X, returns, close_prices


def build_label_sets(next_return: pd.Series) -> Dict[str, pd.Series]:
    labels = {
        "next_day": (next_return > 0).astype(int),
        "future_3d": (next_return.rolling(window=3, min_periods=3).sum().shift(-2) > 0).astype(float),
        "future_5d": (next_return.rolling(window=5, min_periods=5).sum().shift(-4) > 0).astype(float),
        "future_5d_gt_0.5pct": (next_return.rolling(window=5, min_periods=5).sum().shift(-4) > 0.005).astype(float),
    }
    return {name: series.dropna().astype(int) for name, series in labels.items()}


def evaluate_classifier(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_test, y_score)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def run_random_forest_diagnostics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[List[float], Dict[str, float], List[Tuple[str, float]]]:
    tscv = TimeSeriesSplit(n_splits=5)
    fold_aucs: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        fold_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_score = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_score)
        fold_aucs.append(float(fold_auc))
        print(f"  RF fold {fold_idx} AUC: {fold_auc:.6f}")

    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    test_metrics = evaluate_classifier(final_model, X_train, y_train, X_test, y_test)
    importances = sorted(
        zip(X_train.columns.tolist(), final_model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    return fold_aucs, test_metrics, importances


def run_logistic_regression_diagnostics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ]
    )
    return evaluate_classifier(model, X_train, y_train, X_test, y_test)


def diagnose_symbol(symbol: str) -> None:
    print(f"\n=== {symbol} ===")
    X, next_return, _ = build_symbol_dataset(symbol)
    label_sets = build_label_sets(next_return)

    print(f"Feature dimension: {X.shape[1]}")
    summary_rows = []

    for label_name, labels in label_sets.items():
        aligned_index = X.index.intersection(labels.index)
        X_aligned = X.loc[aligned_index]
        y_aligned = labels.loc[aligned_index]

        split_idx = int(len(X_aligned) * 0.8)
        X_train, X_test = X_aligned.iloc[:split_idx], X_aligned.iloc[split_idx:]
        y_train, y_test = y_aligned.iloc[:split_idx], y_aligned.iloc[split_idx:]

        print(f"\n  [{label_name}]")
        print(
            f"  Samples train/test: {len(X_train)}/{len(X_test)}, "
            f"positive_ratio={y_aligned.mean():.6f}"
        )

        fold_aucs, rf_test_metrics, importances = run_random_forest_diagnostics(X_train, y_train, X_test, y_test)
        print(f"  RF mean CV AUC: {np.mean(fold_aucs):.6f}")
        print(
            "  RF test metrics: "
            f"AUC={rf_test_metrics['AUC']:.6f}, "
            f"BalancedAcc={rf_test_metrics['Balanced_Accuracy']:.6f}, "
            f"F1={rf_test_metrics['F1']:.6f}"
        )
        print("  RF top 15 feature importances:")
        for feature_name, importance in importances:
            print(f"    {feature_name}: {importance:.6f}")

        lr_test_metrics = run_logistic_regression_diagnostics(X_train, y_train, X_test, y_test)
        print(f"  LR test AUC: {lr_test_metrics['AUC']:.6f}")

        summary_rows.append(
            {
                "label": label_name,
                "positive_ratio": float(y_aligned.mean()),
                "RF_CV_AUC": float(np.mean(fold_aucs)),
                "RF_Test_AUC": float(rf_test_metrics["AUC"]),
                "LR_Test_AUC": float(lr_test_metrics["AUC"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(f"\n  Summary table for {symbol}:")
    print(summary_df.to_string(index=False))


def main() -> None:
    print("Running feature predictive-power diagnostics")
    for symbol in ["CSI100", "CSI300"]:
        diagnose_symbol(symbol)


if __name__ == "__main__":
    main()
