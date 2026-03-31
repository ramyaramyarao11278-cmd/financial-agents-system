from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from ensemble_model import build_classifier_features, rolling_auto_arima_predict, train_transformer_regressor
from utils.classifier_model import create_classifier_model
from utils.evaluation import find_best_threshold, regression_metrics
from utils.experiment_utils import build_experiment_dataset


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")


def ensure_dirs() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    local_path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    parent_path = os.path.join(os.path.dirname(BASE_DIR), "..", "outputs", name)
    parent_path = os.path.abspath(parent_path)
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    raise FileNotFoundError(name)


def prepare_final_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = os.path.join(OUTPUT_DIR, "final_baseline_comparison.csv")
    ablation_path = os.path.join(OUTPUT_DIR, "final_ablation.csv")
    if os.path.exists(baseline_path) and os.path.exists(ablation_path):
        return pd.read_csv(baseline_path), pd.read_csv(ablation_path)

    baseline_v3 = load_csv("baseline_comparison_v3.csv")
    final_existing = load_csv("final_baseline_comparison.csv")
    final_ablation_existing = load_csv("final_ablation.csv")

    transformer_reg = final_existing.set_index("Model").loc["Transformer_Regression"]
    transformer_cls = final_existing.set_index("Model").loc["Transformer_Classification"]
    ensemble_reg = final_existing.set_index("Model").loc["Ensemble_Regression"]
    ensemble_cls = final_existing.set_index("Model").loc["Ensemble_Classification"]

    rows = baseline_v3.to_dict("records")
    for row in rows:
        if row["Model"] == "Transformer":
            row["R2"] = float(transformer_reg["R2"])
            row["RMSE"] = float(transformer_reg["RMSE"])
            row["MAE"] = float(transformer_reg["MAE"])
            row["AUC"] = float(transformer_cls["AUC"])
            row["Balanced_Acc"] = float(transformer_cls["Balanced_Acc"])
            row["F1"] = float(transformer_cls["F1"])
    rows.append(
        {
            "Model": "Ensemble",
            "R2": float(ensemble_reg["R2"]),
            "RMSE": float(ensemble_reg["RMSE"]),
            "MAE": float(ensemble_reg["MAE"]),
            "AUC": float(ensemble_cls["AUC"]),
            "Balanced_Acc": float(ensemble_cls["Balanced_Acc"]),
            "F1": float(ensemble_cls["F1"]),
        }
    )

    final_baseline = pd.DataFrame(rows)
    final_baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    final_ablation_existing.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    return final_baseline, final_ablation_existing


def fit_csi300_models() -> Dict[str, object]:
    dataset = build_experiment_dataset(symbol="CSI300")

    lr_reg = LinearRegression()
    X_train_flat = dataset.X_train.reshape(len(dataset.X_train), -1)
    X_test_flat = dataset.X_test.reshape(len(dataset.X_test), -1)
    X_val_flat = dataset.X_val.reshape(len(dataset.X_val), -1)
    lr_reg.fit(X_train_flat, dataset.y_return_train)
    lr_test_prob = 1.0 / (1.0 + np.exp(-lr_reg.predict(X_test_flat) * 100.0))
    lr_val_prob = 1.0 / (1.0 + np.exp(-lr_reg.predict(X_val_flat) * 100.0))

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, dataset.y_direction_train)
    rf_val_prob = rf.predict_proba(X_val_flat)[:, 1]
    rf_test_prob = rf.predict_proba(X_test_flat)[:, 1]

    reg_model, _, _, _, _ = train_transformer_regressor(dataset)

    cls_model = create_classifier_model(sequence_length=20, num_features=dataset.X_train.shape[-1])
    cls_model.train(
        dataset.X_train,
        dataset.y_direction_train,
        X_val=dataset.X_val,
        y_val=dataset.y_direction_val,
        epochs=300,
        batch_size=min(32, len(dataset.X_train)),
    )
    transformer_val_prob = cls_model.predict_proba(dataset.X_val)
    transformer_test_prob = cls_model.predict_proba(dataset.X_test)

    features = build_classifier_features(dataset, reg_model)
    fused_lr = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", random_state=42)
    gbt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    fused_lr.fit(features["train_fused"], dataset.y_direction_train)
    gbt.fit(features["train_fused"], dataset.y_direction_train)
    ensemble_val_prob = 0.5 * fused_lr.predict_proba(features["val_fused"])[:, 1] + 0.5 * gbt.predict_proba(features["val_fused"])[:, 1]
    ensemble_test_prob = 0.5 * fused_lr.predict_proba(features["test_fused"])[:, 1] + 0.5 * gbt.predict_proba(features["test_fused"])[:, 1]

    arima_test = rolling_auto_arima_predict(
        np.concatenate([dataset.y_return_train, dataset.y_return_val]),
        dataset.y_return_test,
    )
    transformer_test = reg_model.predict(dataset.X_test)
    arima_val = rolling_auto_arima_predict(dataset.y_return_train, dataset.y_return_val)
    transformer_val = reg_model.predict(dataset.X_val)
    best_alpha = 0.0
    best_r2 = -np.inf
    for alpha in np.arange(0.0, 1.0001, 0.05):
        val_pred = alpha * arima_val + (1.0 - alpha) * transformer_val
        r2 = regression_metrics(dataset.y_return_val, val_pred)["R2"]
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = float(alpha)
    ensemble_reg_test = best_alpha * arima_test + (1.0 - best_alpha) * transformer_test

    return {
        "dataset": dataset,
        "arima_test": arima_test,
        "transformer_test": transformer_test,
        "ensemble_reg_test": ensemble_reg_test,
        "lr_val_prob": lr_val_prob,
        "lr_test_prob": lr_test_prob,
        "rf_val_prob": rf_val_prob,
        "rf_test_prob": rf_test_prob,
        "transformer_val_prob": transformer_val_prob,
        "transformer_test_prob": transformer_test_prob,
        "ensemble_val_prob": ensemble_val_prob,
        "ensemble_test_prob": ensemble_test_prob,
    }


def plot_regression_comparison(fit_result: Dict[str, object]) -> None:
    dataset = fit_result["dataset"]
    dates = pd.to_datetime(dataset.test_dates)
    actual = dataset.y_return_test
    arima_pred = fit_result["arima_test"]
    transformer_pred = fit_result["transformer_test"]
    ensemble_pred = fit_result["ensemble_reg_test"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(dates, actual, label="真实3日收益率", linewidth=1.4, color="#222222")
    ax.plot(dates, arima_pred, label=f"ARIMA 预测 (R²={regression_metrics(actual, arima_pred)['R2']:.4f})", linewidth=1.1)
    ax.plot(dates, transformer_pred, label=f"Transformer 预测 (R²={regression_metrics(actual, transformer_pred)['R2']:.4f})", linewidth=1.1)
    ax.plot(dates, ensemble_pred, label=f"Ensemble 预测 (R²={regression_metrics(actual, ensemble_pred)['R2']:.4f})", linewidth=1.3)
    ax.set_title("CSI300测试集回归结果对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("3日累计收益率")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(os.path.join(FIGURE_DIR, "regression_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_classification_roc(fit_result: Dict[str, object]) -> None:
    dataset = fit_result["dataset"]
    roc_inputs = {
        "LR": (fit_result["lr_test_prob"], fit_result["lr_val_prob"]),
        "RF": (fit_result["rf_test_prob"], fit_result["rf_val_prob"]),
        "Transformer 分类": (fit_result["transformer_test_prob"], fit_result["transformer_val_prob"]),
        "Ensemble 分类": (fit_result["ensemble_test_prob"], fit_result["ensemble_val_prob"]),
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for label, (test_prob, _) in roc_inputs.items():
        auc_score = roc_auc_score(dataset.y_direction_test, test_prob)
        fpr, tpr, _ = roc_curve(dataset.y_direction_test, test_prob)
        ax.plot(fpr, tpr, linewidth=1.3, label=f"{label} (AUC={auc_score:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_title("CSI300测试集分类ROC曲线")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真正率")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(FIGURE_DIR, "classification_roc.png"), bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bar(final_ablation: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    configs = final_ablation["Config"].tolist()
    r2_values = final_ablation["R2"].astype(float).to_numpy()
    auc_values = final_ablation["AUC"].astype(float).to_numpy()
    x = np.arange(len(configs))

    bars = ax1.bar(x, r2_values, color="#4C78A8", width=0.55, label="R²")
    ax1.set_xlabel("配置")
    ax1.set_ylabel("R²", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)

    ax2 = ax1.twinx()
    ax2.plot(x, auc_values, color="#E45756", marker="o", linewidth=2, label="AUC")
    ax2.set_ylabel("AUC", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")

    ax1.set_title("消融实验结果对比")
    ax1.legend(handles=[bars, ax2.lines[0]], labels=["R²", "AUC"], loc="upper left")
    fig.savefig(os.path.join(FIGURE_DIR, "ablation_bar.png"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(fit_result: Dict[str, object]) -> None:
    dataset = fit_result["dataset"]
    threshold, _, _ = find_best_threshold(dataset.y_direction_val, fit_result["lr_val_prob"])
    pred_label = (fit_result["lr_test_prob"] >= threshold).astype(int)
    cm = confusion_matrix(dataset.y_direction_test, pred_label)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("CSI300上LR分类模型混淆矩阵")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["下跌", "上涨"])
    ax.set_yticklabels(["下跌", "上涨"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)


def plot_sentiment_contribution(final_ablation: pd.DataFrame) -> None:
    subset = final_ablation.set_index("Config").loc[["price_only", "full_model", "sentiment_only"], "Ensemble_AUC"].astype(float)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(subset.index.tolist(), subset.to_numpy(), color=["#72B7B2", "#F58518", "#54A24B"])
    ax.set_title("情绪信息对融合分类性能的贡献")
    ax.set_xlabel("配置")
    ax.set_ylabel("Ensemble_AUC")
    for bar, value in zip(bars, subset.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center", va="bottom")
    fig.savefig(os.path.join(FIGURE_DIR, "sentiment_contribution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_radar(final_baseline: pd.DataFrame) -> None:
    subset = final_baseline.set_index("Model").loc[["LR", "ARIMA", "Transformer", "Ensemble"]].copy()
    metrics = ["R2", "RMSE", "MAE", "AUC", "F1"]
    data = subset[metrics].astype(float)

    def norm_positive(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - series.min()) / span

    def norm_negative(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series.max() - series) / span

    radar_df = pd.DataFrame(index=data.index)
    radar_df["R²"] = norm_positive(data["R2"])
    radar_df["RMSE(反向)"] = norm_negative(data["RMSE"])
    radar_df["MAE(反向)"] = norm_negative(data["MAE"])
    radar_df["AUC"] = norm_positive(data["AUC"])
    radar_df["F1"] = norm_positive(data["F1"])

    labels = radar_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, polar=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for color, (model_name, row) in zip(colors, radar_df.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("模型综合性能雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10))
    fig.savefig(os.path.join(FIGURE_DIR, "model_comparison_radar.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    final_baseline, final_ablation = prepare_final_tables()
    fit_result = fit_csi300_models()
    plot_regression_comparison(fit_result)
    plot_classification_roc(fit_result)
    plot_ablation_bar(final_ablation)
    plot_confusion_matrix(fit_result)
    plot_sentiment_contribution(final_ablation)
    plot_model_comparison_radar(final_baseline)
    print("Saved 6 figures to outputs/figures")


if __name__ == "__main__":
    main()
