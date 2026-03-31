from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_output_csv(filename: str) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path}")
    return pd.read_csv(path)


def annotate_bars(ax, bars, precision: int = 3) -> None:
    y_min, y_max = ax.get_ylim()
    span = (y_max - y_min) or 1.0

    for bar in bars:
        value = float(bar.get_height())
        x_pos = bar.get_x() + bar.get_width() / 2
        if value >= 0:
            y_pos = value + span * 0.02
            va = "bottom"
        else:
            y_pos = value - span * 0.03
            va = "top"
        ax.text(x_pos, y_pos, f"{value:.{precision}f}", ha="center", va=va, fontsize=8)


def plot_baseline_overview() -> Path:
    df = load_output_csv("final_baseline_comparison.csv")
    order = ["Naive", "LR", "ARIMA", "LSTM", "GRU", "RF", "Transformer", "Ensemble"]
    df = df.set_index("Model").loc[order].reset_index()

    colors = [
        "#9CA3AF",
        "#4C78A8",
        "#F58518",
        "#9CA3AF",
        "#9CA3AF",
        "#9CA3AF",
        "#54A24B",
        "#E45756",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    bars_r2 = axes[0].bar(df["Model"], df["R2"].astype(float), color=colors)
    axes[0].axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    axes[0].set_title("CSI300基线模型回归表现")
    axes[0].set_ylabel("R^2")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=25)
    annotate_bars(axes[0], bars_r2)

    bars_auc = axes[1].bar(df["Model"], df["AUC"].astype(float), color=colors)
    axes[1].axhline(0.5, color="#666666", linewidth=0.9, linestyle="--", label="随机水平")
    axes[1].set_title("CSI300基线模型分类表现")
    axes[1].set_ylabel("AUC")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, loc="upper right")
    annotate_bars(axes[1], bars_auc)

    fig.suptitle("基线模型总体比较", fontsize=14)
    fig.tight_layout()

    output_path = FIGURE_DIR / "baseline_overview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ablation_overview() -> Path:
    df = load_output_csv("final_ablation.csv")
    order = ["price_only", "sentiment_only", "no_lag_return", "full_model"]
    df = df.set_index("Config").loc[order].reset_index()

    colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    bars_r2 = axes[0].bar(df["Config"], df["R2"].astype(float), color=colors)
    axes[0].axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    axes[0].set_title("消融实验回归表现")
    axes[0].set_ylabel("R^2")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=18)
    annotate_bars(axes[0], bars_r2)

    bars_auc = axes[1].bar(df["Config"], df["Ensemble_AUC"].astype(float), color=colors)
    axes[1].axhline(0.5, color="#666666", linewidth=0.9, linestyle="--", label="随机水平")
    axes[1].set_title("消融实验融合分类表现")
    axes[1].set_ylabel("Ensemble AUC")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(frameon=False, loc="upper right")
    annotate_bars(axes[1], bars_auc)

    fig.suptitle("消融实验结果概览", fontsize=14)
    fig.tight_layout()

    output_path = FIGURE_DIR / "ablation_overview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_close_comparison(symbol: str) -> Path:
    df = load_output_csv(f"{symbol}_prediction_results.csv")
    dates = pd.to_datetime(df["Date"])

    actual_col = "Actual_Close" if "Actual_Close" in df.columns else "Actual_Price"
    predicted_col = "Predicted_Close" if "Predicted_Close" in df.columns else "Predicted_Price"

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.plot(dates, df[actual_col].astype(float), color="#1F2937", linewidth=1.5, label="真实收盘价")
    ax.plot(
        dates,
        df[predicted_col].astype(float),
        color="#F58518",
        linewidth=1.5,
        linestyle="--",
        label="预测收盘价",
    )

    ax.set_title(f"{symbol} 测试集收盘价对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()

    output_path = FIGURE_DIR / f"{symbol}_close_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    ensure_dirs()

    generated = [
        plot_baseline_overview(),
        plot_ablation_overview(),
        plot_close_comparison("CSI100"),
        plot_close_comparison("CSI300"),
    ]

    print("Generated paper-ready figures:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
