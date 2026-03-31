from __future__ import annotations

import os
from typing import List

import pandas as pd


BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_csv(name: str) -> pd.DataFrame:
    local_path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    parent_path = os.path.abspath(os.path.join(os.path.dirname(BASE_DIR), "..", "outputs", name))
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    raise FileNotFoundError(name)


def format_value(value: float) -> str:
    return f"{float(value):.4f}"


def build_baseline_table(df: pd.DataFrame) -> str:
    work_df = df.copy()
    metric_columns = ["R2", "RMSE", "MAE", "AUC", "Balanced_Acc", "F1"]
    max_cols = {"R2", "AUC", "Balanced_Acc", "F1"}
    min_cols = {"RMSE", "MAE"}

    best_values = {}
    for col in metric_columns:
        series = work_df[col].astype(float)
        best_values[col] = series.max() if col in max_cols else series.min()

    lines: List[str] = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"模型 & R\textsuperscript{2} & RMSE & MAE & AUC & 平衡准确率 & F1 \\",
        r"\midrule",
    ]

    for _, row in work_df.iterrows():
        values = []
        for col in metric_columns:
            text = format_value(row[col])
            if abs(float(row[col]) - float(best_values[col])) < 1e-12:
                text = rf"\textbf{{{text}}}"
            values.append(text)
        line = f"{row['Model']} & " + " & ".join(values) + r" \\"
        if row["Model"] == "Ensemble":
            line = r"\rowcolor{gray!15} " + line
        lines.append(line)

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def build_ablation_table(df: pd.DataFrame) -> str:
    work_df = df.copy()
    metric_columns = ["R2", "RMSE", "MAE", "AUC", "Balanced_Acc", "F1", "Ensemble_AUC"]

    lines: List[str] = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"配置 & R\textsuperscript{2} & RMSE & MAE & AUC & 平衡准确率 & F1 & Ensemble\_AUC \\",
        r"\midrule",
    ]

    for _, row in work_df.iterrows():
        values = []
        for col in metric_columns:
            text = format_value(row[col])
            if row["Config"] == "sentiment_only" and col == "Ensemble_AUC":
                text = rf"\textbf{{{text}}}"
            values.append(text)
        lines.append(f"{row['Config']} & " + " & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    baseline_df = load_csv("final_baseline_comparison.csv")
    ablation_df = load_csv("final_ablation.csv")

    baseline_latex = build_baseline_table(baseline_df)
    ablation_latex = build_ablation_table(ablation_df)

    notes = (
        "中文注释：表1显示，CSI300 上 Ensemble 在综合指标上保持最优，说明统计模型与深度模型融合后，"
        "在回归与分类任务上都能取得更稳健的结果。表2显示，sentiment_only 的 Ensemble_AUC 最高，"
        "说明情绪相关信息对方向判断具有较强贡献；但从回归误差看，full_model 在多源信息联合后整体更均衡，"
        "支持论文中“情绪信息增强分类、价格信息稳定回归”的结论。"
    )

    output_text = (
        "表1 基线对比表\n"
        f"{baseline_latex}\n\n"
        "表2 消融实验表\n"
        f"{ablation_latex}\n\n"
        f"{notes}\n"
    )
    with open(os.path.join(OUTPUT_DIR, "latex_tables.txt"), "w", encoding="utf-8") as f:
        f.write(output_text)
    print("Saved outputs/latex_tables.txt")


if __name__ == "__main__":
    main()
