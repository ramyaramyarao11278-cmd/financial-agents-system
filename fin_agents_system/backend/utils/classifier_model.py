#!/usr/bin/env python3
"""
Standalone Transformer classifier for multi-day direction prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import layers, models

from utils.logger import get_logger

logger = get_logger(__name__)


class TransformerClassifierModel:
    def __init__(
        self,
        sequence_length: int = 20,
        num_features: int = 1,
        d_model: int = 32,
        num_heads: int = 2,
        dff: int = 64,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.feature_scaler = RobustScaler()
        self.model: Optional[models.Model] = None
        self.pos_weight: float = 1.0

    def _sinusoidal_position_encoding(self, length: int, depth: int) -> tf.Tensor:
        positions = np.arange(length)[:, np.newaxis]
        div_terms = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth))
        encoding = np.zeros((length, depth), dtype=np.float32)
        encoding[:, 0::2] = np.sin(positions * div_terms)
        encoding[:, 1::2] = np.cos(positions * div_terms)
        return tf.constant(encoding[np.newaxis, ...], dtype=tf.float32)

    def _build_model(self) -> models.Model:
        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name="features")
        x = layers.Dense(self.d_model, name="input_projection")(inputs)
        x = x + self._sinusoidal_position_encoding(self.sequence_length, self.d_model)

        attn_input = layers.LayerNormalization(epsilon=1e-6, name="attn_ln")(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name="attn",
        )(attn_input, attn_input)
        attn_output = layers.Dropout(self.dropout_rate, name="attn_dropout")(attn_output)
        x = layers.Add(name="attn_residual")([x, attn_output])

        ffn_input = layers.LayerNormalization(epsilon=1e-6, name="ffn_ln")(x)
        ffn_output = layers.Dense(self.dff, activation="gelu", name="ffn_expand")(ffn_input)
        ffn_output = layers.Dropout(self.dropout_rate, name="ffn_dropout")(ffn_output)
        ffn_output = layers.Dense(self.d_model, name="ffn_project")(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate, name="ffn_project_dropout")(ffn_output)
        x = layers.Add(name="ffn_residual")([x, ffn_output])

        x = layers.LayerNormalization(epsilon=1e-6, name="final_layer_norm")(x)
        x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)
        x = layers.Dense(16, activation="relu", name="classifier_hidden")(x)
        x = layers.Dropout(self.dropout_rate, name="classifier_dropout")(x)
        outputs = layers.Dense(1, activation=None, name="classifier_head")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="transformer_classifier")
        model.summary(print_fn=logger.info)
        logger.info("Classifier Transformer total params: %s", model.count_params())
        return model

    def _ensure_model(self, num_features: int) -> None:
        if self.model is None or self.num_features != num_features:
            self.num_features = num_features
            self.model = self._build_model()

    def fit_feature_scaler(self, X_train: np.ndarray) -> None:
        flat_train = X_train.reshape(-1, X_train.shape[-1])
        self.feature_scaler.fit(flat_train)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        flat = X.reshape(-1, X.shape[-1])
        scaled = self.feature_scaler.transform(flat)
        return scaled.reshape(X.shape).astype(np.float32)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: int = 32,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        tf.keras.utils.set_random_seed(42)
        np.random.seed(42)

        self._ensure_model(X_train.shape[-1])
        self.fit_feature_scaler(X_train)

        X_train_scaled = self.transform_features(X_train)
        y_train = np.asarray(y_train).astype(np.float32)

        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        self.pos_weight = float(negatives / max(positives, 1.0))

        logger.info("Classifier pos_weight=%s", self.pos_weight)

        pos_weight = tf.constant(self.pos_weight, dtype=tf.float32)

        def weighted_bce_with_logits(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32)
            return tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
            )

        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=max(int(np.ceil(len(X_train_scaled) / batch_size)) * 15, 1),
            t_mul=2.0,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=5e-4,
            clipnorm=1.0,
        )

        class DegeneratePredictionMonitor(tf.keras.callbacks.Callback):
            def __init__(self, X_val_array: np.ndarray):
                super().__init__()
                self.X_val_array = X_val_array
                self.degenerate_epochs = 0

            def on_epoch_end(self, epoch, logs=None):
                val_logits = self.model.predict(self.X_val_array, verbose=0).reshape(-1)
                val_prob = tf.sigmoid(val_logits).numpy()
                positive_ratio = float(np.mean(val_prob > 0.5))
                if positive_ratio > 0.8 or positive_ratio < 0.2:
                    self.degenerate_epochs += 1
                else:
                    self.degenerate_epochs = 0
                if self.degenerate_epochs >= 5:
                    logger.warning("模型退化为单一预测 | epoch=%s predicted_up_ratio=%.4f", epoch + 1, positive_ratio)

        self.model.compile(
            optimizer=optimizer,
            loss=weighted_bce_with_logits,
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy"),
            ],
        )

        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc" if X_val is not None and y_val is not None else "auc",
                patience=30,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            )
        ]
        if model_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_path,
                    monitor="val_auc" if X_val is not None and y_val is not None else "auc",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="max",
                    verbose=0,
                )
            )

        fit_kwargs: Dict[str, Any] = {
            "x": X_train_scaled,
            "y": y_train,
            "epochs": epochs,
            "batch_size": batch_size,
            "shuffle": True,
            "callbacks": callbacks,
            "verbose": 0,
        }
        if X_val is not None and y_val is not None:
            X_val_scaled = self.transform_features(X_val)
            y_val_array = np.asarray(y_val).astype(np.float32)
            callbacks.append(DegeneratePredictionMonitor(X_val_scaled))
            fit_kwargs["validation_data"] = (self.transform_features(X_val), np.asarray(y_val).astype(np.float32))

        history = self.model.fit(**fit_kwargs)
        return history.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.transform_features(X)
        logits = self.model.predict(X_scaled, verbose=0).reshape(-1)
        return tf.sigmoid(logits).numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def diagnose_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        train_prob = self.predict_proba(X_train)
        test_prob = self.predict_proba(X_test)
        y_train = np.asarray(y_train).astype(int)
        y_test = np.asarray(y_test).astype(int)

        train_positive_ratio = float(y_train.mean())
        test_positive_ratio = float(y_test.mean())
        probability_stats = {
            "mean": float(np.mean(test_prob)),
            "std": float(np.std(test_prob)),
            "min": float(np.min(test_prob)),
            "max": float(np.max(test_prob)),
            "median": float(np.median(test_prob)),
            "gt_0.5_ratio": float(np.mean(test_prob > 0.5)),
        }
        auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5
        flipped_prob = 1.0 - test_prob
        flipped_auc = float(roc_auc_score(y_test, flipped_prob)) if len(np.unique(y_test)) > 1 else 0.5
        flipped_pred = (flipped_prob >= threshold).astype(int)

        label_consistency = bool(np.array_equal((y_test > 0).astype(int), y_test))

        logger.info("Classifier probability stats on test set: %s", probability_stats)
        logger.info(
            "Classifier positive ratios | train=%.4f test=%.4f pos_weight=%.4f",
            train_positive_ratio,
            test_positive_ratio,
            self.pos_weight,
        )
        logger.info(
            "Label consistency check | train_matches_return_sign=%s test_matches_return_sign=%s",
            bool(np.array_equal((y_train > 0).astype(int), y_train)),
            label_consistency,
        )
        logger.info(
            "Current pos_weight formula uses negatives/positives. negatives/positives=%.4f",
            float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)),
        )
        logger.info("Original test AUC=%.6f | Flipped test AUC=%.6f", auc, flipped_auc)
        logger.info(
            "Flipped metrics | Balanced_Acc=%.6f F1=%.6f",
            float(balanced_accuracy_score(y_test, flipped_pred)),
            float(f1_score(y_test, flipped_pred, zero_division=0)),
        )

        return {
            "test_auc": auc,
            "flipped_test_auc": flipped_auc,
            "train_positive_ratio": train_positive_ratio,
            "test_positive_ratio": test_positive_ratio,
            "prob_mean": probability_stats["mean"],
        }


def create_classifier_model(**kwargs) -> TransformerClassifierModel:
    return TransformerClassifierModel(**kwargs)
