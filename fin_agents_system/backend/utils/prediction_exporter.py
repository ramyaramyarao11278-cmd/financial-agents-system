import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.evaluation import confusion_matrix_values, export_metrics_to_csv


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def export_prediction_csv(
    symbol,
    dates,
    actual_returns,
    predicted_returns,
    actual_prices,
    predicted_prices,
    direction_prob,
    direction_label,
    output_dir="outputs",
    classification_threshold=None,
):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{symbol}_prediction_results.csv")
    metrics_path = os.path.join(output_dir, f"{symbol}_prediction_metrics.csv")

    direction_prob = np.asarray(direction_prob, dtype=float)
    direction_label = np.asarray(direction_label).astype(int)
    actual_direction = (np.asarray(actual_returns, dtype=float) > 0).astype(int)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Actual_Return": actual_returns,
            "Predicted_Return": predicted_returns,
            "Actual_Price": actual_prices,
            "Predicted_Price": predicted_prices,
            "Actual_Direction": actual_direction,
            "Direction_Prob": direction_prob,
            "Direction_Label": direction_label,
        }
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    export_metrics_to_csv(
        actual_prices=actual_returns,
        predicted_prices=predicted_returns,
        path=metrics_path,
        symbol=symbol,
        y_true_cls=actual_direction,
        y_pred_cls=direction_label,
        y_score=direction_prob,
        threshold=classification_threshold,
    )

    return csv_path, metrics_path


def export_return_plot(symbol, dates, actual, predicted, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{symbol}_return_curve.png")

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="真实收益率")
    plt.plot(dates, predicted, label="预测收益率")
    plt.legend()
    plt.title(f"{symbol} 收益率预测")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def export_price_plot(symbol, dates, actual, predicted, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{symbol}_price_curve.png")

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="真实价格")
    plt.plot(dates, predicted, label="预测价格")
    plt.legend()
    plt.title(f"{symbol} 价格路径对比")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def export_confusion_matrix_plot(symbol, y_true, y_pred, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{symbol}_confusion_matrix.png")

    matrix = confusion_matrix_values(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["跌", "涨"])
    ax.set_yticklabels(["跌", "涨"])
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"{symbol} 混淆矩阵")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def export_roc_plot(symbol, fpr, tpr, auc_score, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{symbol}_roc_curve.png")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title(f"{symbol} ROC 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path
