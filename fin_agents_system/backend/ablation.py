from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.classifier_model import create_classifier_model
from utils.evaluation import regression_metrics
from utils.experiment_utils import build_experiment_dataset, classification_metrics_from_probabilities, combine_metrics
from utils.transformer_model import create_transformer_model


def get_feature_subset(mode: str):
    base_dataset = build_experiment_dataset(symbol="CSI300")
    names = base_dataset.feature_names
    if mode == "price_only":
        selected = [name for name in names if not (name.startswith("sentiment_") or name.startswith("news_"))]
    elif mode == "sentiment_only":
        selected = [name for name in names if name.startswith("sentiment_") or name.startswith("news_")]
        selected += [name for name in ["lag_return_1", "lag_return_2", "lag_return_3", "Close"] if name in names]
        selected = list(dict.fromkeys(selected))
    elif mode == "no_lag_return":
        selected = [name for name in names if name not in {"lag_return_1", "lag_return_2", "lag_return_3"}]
    else:
        selected = names
    return selected


def run_configuration(mode: str):
    dataset = build_experiment_dataset(symbol="CSI300", feature_subset=get_feature_subset(mode))
    reg_model = create_transformer_model(sequence_length=20, num_features=dataset.X_train.shape[-1])
    cls_model = create_classifier_model(sequence_length=20, num_features=dataset.X_train.shape[-1])

    reg_history = reg_model.train(
        dataset.X_train,
        dataset.y_return_train,
        X_val=dataset.X_val,
        y_val=dataset.y_return_val,
        epochs=300,
        batch_size=min(32, len(dataset.X_train)),
    )
    cls_history = cls_model.train(
        dataset.X_train,
        dataset.y_direction_train,
        X_val=dataset.X_val,
        y_val=dataset.y_direction_val,
        epochs=300,
        batch_size=min(32, len(dataset.X_train)),
    )

    reg_train_pred = reg_model.predict(dataset.X_train)
    reg_pred = reg_model.predict(dataset.X_test)
    train_reg_metrics = regression_metrics(dataset.y_return_train, reg_train_pred)
    test_reg_metrics = regression_metrics(dataset.y_return_test, reg_pred)
    val_prob = cls_model.predict_proba(dataset.X_val)
    train_prob = cls_model.predict_proba(dataset.X_train)
    test_prob = cls_model.predict_proba(dataset.X_test)
    metrics = combine_metrics(
        dataset.y_return_test,
        reg_pred,
        dataset.y_direction_test,
        test_prob,
        dataset.y_direction_val,
        val_prob,
    )
    metrics["Feature_Count"] = len(dataset.feature_names)
    print(f"{mode} feature count: {len(dataset.feature_names)}")
    return {
        "metrics": metrics,
        "train_reg_metrics": train_reg_metrics,
        "test_reg_metrics": test_reg_metrics,
        "train_cls_metrics": classification_metrics_from_probabilities(
            dataset.y_direction_train,
            train_prob,
            dataset.y_direction_val,
            val_prob,
        ),
        "test_cls_metrics": classification_metrics_from_probabilities(
            dataset.y_direction_test,
            test_prob,
            dataset.y_direction_val,
            val_prob,
        ),
        "reg_epochs": len(reg_history.get("loss", [])),
        "cls_epochs": len(cls_history.get("loss", [])),
        "dataset": dataset,
    }


def main() -> None:
    configs = ["price_only", "sentiment_only", "no_lag_return", "full_model"]
    rows = []
    diagnostics = {}
    for config in configs:
        result = run_configuration(config)
        metrics = result["metrics"]
        rows.append({"Config": config, **metrics})
        diagnostics[config] = result

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "ablation_results_v3.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    full_auc = float(df.loc[df["Config"] == "full_model", "AUC"].iloc[0])
    price_auc = float(df.loc[df["Config"] == "price_only", "AUC"].iloc[0])
    sentiment_auc = float(df.loc[df["Config"] == "sentiment_only", "AUC"].iloc[0])
    no_lag_auc = float(df.loc[df["Config"] == "no_lag_return", "AUC"].iloc[0])
    print(
        "\nAblation summary: "
        f"emotion contribution={full_auc - price_auc:.4f}, "
        f"technical contribution={full_auc - sentiment_auc:.4f}, "
        f"lag return contribution={full_auc - no_lag_auc:.4f}"
    )

    print("\n[Ablation diagnostics]")
    for config in ["price_only", "full_model"]:
        diag = diagnostics[config]
        print(
            f"{config}: train_R2={diag['train_reg_metrics']['R2']:.6f}, "
            f"test_R2={diag['test_reg_metrics']['R2']:.6f}, "
            f"train_AUC={diag['train_cls_metrics']['AUC']:.6f}, "
            f"test_AUC={diag['test_cls_metrics']['AUC']:.6f}, "
            f"reg_epochs={diag['reg_epochs']}, cls_epochs={diag['cls_epochs']}"
        )

    full_dataset = diagnostics["full_model"]["dataset"]
    flat_full = full_dataset.X_train.reshape(-1, full_dataset.X_train.shape[-1])
    corr_df = pd.DataFrame(flat_full, columns=full_dataset.feature_names).corr().abs()
    high_corr_pairs = []
    for row_idx in range(len(corr_df.columns)):
        for col_idx in range(row_idx + 1, len(corr_df.columns)):
            corr_value = float(corr_df.iloc[row_idx, col_idx])
            if corr_value > 0.9:
                high_corr_pairs.append(
                    (
                        corr_df.index[row_idx],
                        corr_df.columns[col_idx],
                        corr_value,
                    )
                )
    high_corr_pairs = sorted(high_corr_pairs, key=lambda item: item[2], reverse=True)
    print("full_model highly correlated feature pairs (|corr| > 0.9):")
    if high_corr_pairs:
        for left, right, corr_value in high_corr_pairs:
            print(f"{left} <-> {right}: {corr_value:.4f}")
    else:
        print("None")


if __name__ == "__main__":
    main()
