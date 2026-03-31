from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from utils.classifier_model import create_classifier_model
from utils.evaluation import regression_metrics
from utils.experiment_utils import build_experiment_dataset, classification_metrics_from_probabilities, combine_metrics
from utils.transformer_model import create_transformer_model

try:
    from pmdarima import auto_arima  # type: ignore
except Exception:
    auto_arima = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None


class TorchSequenceModel(nn.Module):
    def __init__(self, cell_type: str, input_size: int, task: str):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.task = task
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.15,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        pooled = out[:, -1, :]
        return self.head(pooled).squeeze(-1)


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    epochs: int = 100,
    patience: int = 15,
) -> nn.Module:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    if task == "regression":
        criterion = nn.MSELoss()
        mode = "min"
    else:
        positives = float((y_train > 0.5).sum())
        negatives = float((y_train <= 0.5).sum())
        pos_weight = torch.tensor([max(negatives / max(positives, 1.0), 1.0)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mode = "max"

    best_state = None
    best_metric = np.inf if mode == "min" else -np.inf
    wait = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            if task == "regression":
                metric = float(criterion(val_output, y_val_tensor).item())
                improved = metric < best_metric - 1e-5
            else:
                val_prob = torch.sigmoid(val_output).cpu().numpy()
                metric = float(classification_metrics_from_probabilities(y_val, val_prob)["AUC"])
                improved = metric > best_metric + 1e-5

        if improved:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_arima_forecaster(history: list[float]):
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
        raise RuntimeError("ARIMA baseline requires either pmdarima or statsmodels.")

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
    if hasattr(model, "predict") and auto_arima is not None:
        return np.asarray(model.predict(n_periods=n_periods), dtype=float)
    return np.asarray(model.forecast(steps=n_periods), dtype=float)


def evaluate_torch_baseline(cell_type: str, dataset) -> Dict[str, float]:
    reg_model = train_torch_model(
        TorchSequenceModel(cell_type, dataset.X_train.shape[-1], "regression"),
        dataset.X_train,
        dataset.y_return_train,
        dataset.X_val,
        dataset.y_return_val,
        task="regression",
    )
    cls_model = train_torch_model(
        TorchSequenceModel(cell_type, dataset.X_train.shape[-1], "classification"),
        dataset.X_train,
        dataset.y_direction_train,
        dataset.X_val,
        dataset.y_direction_val,
        task="classification",
    )
    with torch.no_grad():
        reg_pred = reg_model(torch.tensor(dataset.X_test, dtype=torch.float32)).cpu().numpy()
        val_prob = torch.sigmoid(cls_model(torch.tensor(dataset.X_val, dtype=torch.float32))).cpu().numpy()
        test_prob = torch.sigmoid(cls_model(torch.tensor(dataset.X_test, dtype=torch.float32))).cpu().numpy()
    return combine_metrics(
        dataset.y_return_test,
        reg_pred,
        dataset.y_direction_test,
        test_prob,
        dataset.y_direction_val,
        val_prob,
    )


def evaluate_naive(dataset) -> Dict[str, float]:
    reg_pred = dataset.past_horizon_return_test
    cls_prob = (dataset.past_horizon_return_test > 0).astype(float)
    return combine_metrics(dataset.y_return_test, reg_pred, dataset.y_direction_test, cls_prob)


def evaluate_linear_regression(dataset) -> Dict[str, float]:
    X_train = dataset.X_train.reshape(len(dataset.X_train), -1)
    X_test = dataset.X_test.reshape(len(dataset.X_test), -1)
    model = LinearRegression()
    model.fit(X_train, dataset.y_return_train)
    reg_pred = model.predict(X_test)
    cls_prob = 1.0 / (1.0 + np.exp(-reg_pred * 100.0))
    return combine_metrics(dataset.y_return_test, reg_pred, dataset.y_direction_test, cls_prob)


def evaluate_arima(dataset) -> Dict[str, float]:
    train_series = pd.Series(dataset.y_return_train)
    test_series = pd.Series(dataset.y_return_test)
    forecasts = []
    history = train_series.tolist()
    limit = len(test_series)
    refit_interval = 50
    mode = "rolling_auto_arima_refit_every_50" if auto_arima is not None else "rolling_statsmodels_arima_refit_every_50"

    for start in range(0, limit, refit_interval):
        stop = min(start + refit_interval, limit)
        model = fit_arima_forecaster(history)
        block_forecast = forecast_arima(model, stop - start)
        forecasts.extend([float(value) for value in block_forecast])
        for idx in range(start, stop):
            history.append(float(test_series.iloc[idx]))

    reg_true = test_series.iloc[:limit].to_numpy()
    reg_pred = np.asarray(forecasts, dtype=float)
    cls_prob = 1.0 / (1.0 + np.exp(-reg_pred * 100.0))
    cls_true = dataset.y_direction_test[:limit]
    cls_pred = (cls_prob >= 0.5).astype(int)

    print("\n[ARIMA diagnostics]")
    print(f"mode={mode}")
    print("rolling_prediction_only_uses_train_plus_past_test=True")
    print(f"history_initial_length={len(train_series)}, test_limit={limit}")
    print(f"prediction_scale=multi_day_cumulative_return, evaluation_scale=multi_day_cumulative_return")
    print(f"first_10_pred={np.round(reg_pred[:10], 6).tolist()}")
    print(f"first_10_true={np.round(reg_true[:10], 6).tolist()}")
    print(f"classification_rule=predicted_return_gt_0 -> sigmoid -> threshold_0.5")
    print(f"predicted_positive_ratio={float(cls_pred.mean()):.4f}")
    print(f"true_positive_ratio={float(cls_true.mean()):.4f}")
    print("test_actual_values_are_appended_to_history_after_each_step=True")
    print("future_test_targets_are_not_used_before_their_step=True")
    print(
        "ARIMA stats | "
        f"mean={float(np.mean(reg_pred)):.6f}, std={float(np.std(reg_pred)):.6f}, "
        f"min={float(np.min(reg_pred)):.6f}, max={float(np.max(reg_pred)):.6f}"
    )

    return combine_metrics(reg_true, reg_pred, cls_true, cls_prob)


def evaluate_random_forest(dataset) -> Dict[str, float]:
    X_train = dataset.X_train.reshape(len(dataset.X_train), -1)
    X_val = dataset.X_val.reshape(len(dataset.X_val), -1)
    X_test = dataset.X_test.reshape(len(dataset.X_test), -1)

    reg_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    cls_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    reg_model.fit(X_train, dataset.y_return_train)
    cls_model.fit(X_train, dataset.y_direction_train)

    reg_pred = reg_model.predict(X_test)
    val_prob = cls_model.predict_proba(X_val)[:, 1]
    test_prob = cls_model.predict_proba(X_test)[:, 1]
    return combine_metrics(
        dataset.y_return_test,
        reg_pred,
        dataset.y_direction_test,
        test_prob,
        dataset.y_direction_val,
        val_prob,
    )


def evaluate_transformer_current(dataset) -> Dict[str, float]:
    reg_model = create_transformer_model(sequence_length=20, num_features=dataset.X_train.shape[-1])
    reg_history = reg_model.train(
        dataset.X_train,
        dataset.y_return_train,
        X_val=dataset.X_val,
        y_val=dataset.y_return_val,
        epochs=300,
        batch_size=min(32, len(dataset.X_train)),
    )
    cls_model = create_classifier_model(sequence_length=20, num_features=dataset.X_train.shape[-1])
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
    cls_train_prob = cls_model.predict_proba(dataset.X_train)
    val_prob = cls_model.predict_proba(dataset.X_val)
    test_prob = cls_model.predict_proba(dataset.X_test)
    cls_diag = cls_model.diagnose_predictions(
        dataset.X_train,
        dataset.y_direction_train,
        dataset.X_test,
        dataset.y_direction_test,
    )

    reg_train_metrics = regression_metrics(dataset.y_return_train, reg_train_pred)
    reg_test_metrics = regression_metrics(dataset.y_return_test, reg_pred)
    cls_train_metrics = classification_metrics_from_probabilities(
        dataset.y_direction_train,
        cls_train_prob,
        dataset.y_direction_val,
        val_prob,
    )
    cls_test_metrics = classification_metrics_from_probabilities(
        dataset.y_direction_test,
        test_prob,
        dataset.y_direction_val,
        val_prob,
    )

    reg_params = int(reg_model.model.count_params())
    cls_params = int(cls_model.model.count_params())
    train_samples = int(len(dataset.X_train))

    print("\n[Transformer regression diagnostics]")
    print(
        f"train_R2={reg_train_metrics['R2']:.6f}, train_MSE={reg_train_metrics['MSE']:.6f}, "
        f"test_R2={reg_test_metrics['R2']:.6f}, test_MSE={reg_test_metrics['MSE']:.6f}"
    )
    print(
        f"params={reg_params}, train_samples={train_samples}, params_per_sample={reg_params / max(train_samples, 1):.4f}"
    )
    for idx in range(0, len(reg_history.get('loss', [])), 10):
        epoch_no = idx + 1
        train_loss = reg_history["loss"][idx]
        val_loss = reg_history.get("val_loss", [None] * len(reg_history["loss"]))[idx]
        print(f"epoch={epoch_no}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    print("\n[Transformer classification diagnostics]")
    print(
        f"train_AUC={cls_train_metrics['AUC']:.6f}, train_BalancedAcc={cls_train_metrics['Balanced_Acc']:.6f}, "
        f"test_AUC={cls_test_metrics['AUC']:.6f}, test_BalancedAcc={cls_test_metrics['Balanced_Acc']:.6f}"
    )
    print(
        f"params={cls_params}, train_samples={train_samples}, params_per_sample={cls_params / max(train_samples, 1):.4f}"
    )
    for idx in range(0, len(cls_history.get('loss', [])), 10):
        epoch_no = idx + 1
        train_loss = cls_history["loss"][idx]
        val_loss = cls_history.get("val_loss", [None] * len(cls_history["loss"]))[idx]
        print(f"epoch={epoch_no}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    print(
        f"classifier_direction_check | original_auc={cls_diag['test_auc']:.6f} "
        f"flipped_auc={cls_diag['flipped_test_auc']:.6f}"
    )
    print(
        "Transformer stats | "
        f"pred_mean={float(np.mean(reg_pred)):.6f}, pred_std={float(np.std(reg_pred)):.6f}, "
        f"pred_min={float(np.min(reg_pred)):.6f}, pred_max={float(np.max(reg_pred)):.6f}"
    )
    print(
        "True return stats | "
        f"mean={float(np.mean(dataset.y_return_test)):.6f}, std={float(np.std(dataset.y_return_test)):.6f}, "
        f"min={float(np.min(dataset.y_return_test)):.6f}, max={float(np.max(dataset.y_return_test)):.6f}"
    )

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 5))
    x_axis = np.arange(len(dataset.y_return_test))
    ax.plot(x_axis, dataset.y_return_test, label="真实值", linewidth=1.0)
    ax.plot(x_axis, reg_pred, label="Transformer预测", linewidth=1.0)
    arima_reg_pred = np.full_like(dataset.y_return_test, np.nan, dtype=float)
    try:
        train_series = pd.Series(dataset.y_return_train)
        test_series = pd.Series(dataset.y_return_test)
        history = train_series.tolist()
        forecasts = []
        for start in range(0, len(test_series), 50):
            stop = min(start + 50, len(test_series))
            model = auto_arima(
                history,
                seasonal=False,
                max_p=3,
                max_q=3,
                stepwise=True,
                suppress_warnings=True,
            )
            forecasts.extend([float(value) for value in model.predict(n_periods=stop - start)])
            for idx in range(start, stop):
                history.append(float(test_series.iloc[idx]))
        arima_reg_pred = np.asarray(forecasts, dtype=float)
        ax.plot(x_axis, arima_reg_pred, label="ARIMA预测", linewidth=1.0)
    except Exception as exc:
        print(f"ARIMA diagnostic plot skipped: {exc}")
    ax.set_title("回归诊断对比图")
    ax.set_xlabel("时间")
    ax.set_ylabel("未来3日累计收益率")
    ax.legend()
    ax.grid(alpha=0.2)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(os.path.join("outputs", "regression_diagnostic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return combine_metrics(
        dataset.y_return_test,
        reg_pred,
        dataset.y_direction_test,
        test_prob,
        dataset.y_direction_val,
        val_prob,
    )


def main() -> None:
    dataset = build_experiment_dataset(symbol="CSI300")
    rows = []

    evaluators = {
        "Naive": evaluate_naive,
        "LR": evaluate_linear_regression,
        "ARIMA": evaluate_arima,
        "LSTM": lambda ds: evaluate_torch_baseline("lstm", ds),
        "GRU": lambda ds: evaluate_torch_baseline("gru", ds),
        "RF": evaluate_random_forest,
        "Transformer": evaluate_transformer_current,
    }

    for name, evaluator in evaluators.items():
        metrics = evaluator(dataset)
        rows.append({"Model": name, **metrics})

    df = pd.DataFrame(rows)
    output_path = os.path.join("outputs", "baseline_comparison_v3.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
