#!/usr/bin/env python3
"""
Regression-only Transformer model for multi-day cumulative return forecasting.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import layers, models

from utils.logger import get_logger

logger = get_logger(__name__)


class TransformerTimeSeriesModel:
    """Transformer regression model with a dedicated MLP head."""

    def __init__(
        self,
        sequence_length: int = 20,
        num_features: int = 1,
        d_model: int = 32,
        num_heads: int = 2,
        dff: int = 64,
        num_transformer_layers: int = 1,
        dropout_rate: float = 0.15,
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_transformer_layers = num_transformer_layers
        self.dropout_rate = dropout_rate

        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.model: Optional[models.Model] = None
        self.training_target_std: float = 0.0

    def _sinusoidal_position_encoding(self, length: int, depth: int) -> tf.Tensor:
        positions = np.arange(length)[:, np.newaxis]
        div_terms = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth))
        encoding = np.zeros((length, depth), dtype=np.float32)
        encoding[:, 0::2] = np.sin(positions * div_terms)
        encoding[:, 1::2] = np.cos(positions * div_terms)
        return tf.constant(encoding[np.newaxis, ...], dtype=tf.float32)

    def _transformer_block(self, x: tf.Tensor, layer_idx: int) -> tf.Tensor:
        attn_input = layers.LayerNormalization(epsilon=1e-6, name=f"attn_ln_{layer_idx}")(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f"attn_{layer_idx}",
        )(attn_input, attn_input)
        attn_output = layers.Dropout(self.dropout_rate, name=f"attn_dropout_{layer_idx}")(attn_output)
        x = layers.Add(name=f"attn_residual_{layer_idx}")([x, attn_output])

        ffn_input = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_ln_{layer_idx}")(x)
        ffn_output = layers.Dense(self.dff, activation="gelu", name=f"ffn_expand_{layer_idx}")(ffn_input)
        ffn_output = layers.Dropout(self.dropout_rate, name=f"ffn_dropout_{layer_idx}")(ffn_output)
        ffn_output = layers.Dense(self.d_model, name=f"ffn_project_{layer_idx}")(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate, name=f"ffn_project_dropout_{layer_idx}")(ffn_output)
        return layers.Add(name=f"ffn_residual_{layer_idx}")([x, ffn_output])

    def _build_model(self) -> models.Model:
        logger.info(
            "Building regression Transformer: lookback=%s, features=%s, d_model=%s, layers=%s, heads=%s",
            self.sequence_length,
            self.num_features,
            self.d_model,
            self.num_transformer_layers,
            self.num_heads,
        )

        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name="features")
        x = layers.Dense(self.d_model, name="input_projection")(inputs)
        x = x + self._sinusoidal_position_encoding(self.sequence_length, self.d_model)

        for layer_idx in range(self.num_transformer_layers):
            x = self._transformer_block(x, layer_idx)

        x = layers.LayerNormalization(epsilon=1e-6, name="final_layer_norm")(x)
        x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)
        x = layers.Dense(16, activation="relu", name="regression_head_dense")(x)
        outputs = layers.Dense(1, activation="linear", name="return_head")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="transformer_regression")
        model.summary(print_fn=logger.info)
        logger.info("Regression Transformer total params: %s", model.count_params())
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

    def fit_target_scaler(self, y_train: np.ndarray) -> None:
        self.target_scaler.fit(np.asarray(y_train).reshape(-1, 1))

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(np.asarray(y).reshape(-1, 1)).reshape(-1).astype(np.float32)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).reshape(-1)

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
        self.fit_target_scaler(y_train)
        self.training_target_std = float(np.std(y_train))

        X_train_scaled = self.transform_features(X_train)
        y_train_scaled = self.transform_target(y_train)

        trainable_groups = {
            "encoder": [],
            "head": [],
        }
        for variable in self.model.trainable_variables:
            if "regression_head" in variable.path or "return_head" in variable.path:
                trainable_groups["head"].append(variable)
            else:
                trainable_groups["encoder"].append(variable)

        logger.info(
            "Regression optimizer groups | encoder lr=5e-4 vars=%s | head lr=5e-4 vars=%s",
            [f"{var.path}:{var.shape}" for var in trainable_groups["encoder"]],
            [f"{var.path}:{var.shape}" for var in trainable_groups["head"]],
        )

        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-4,
            first_decay_steps=max(int(np.ceil(len(X_train_scaled) / batch_size)) * 15, 1),
            t_mul=2.0,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=1e-4,
            clipnorm=1.0,
        )

        target_std_constant = tf.constant(self.training_target_std, dtype=tf.float32)

        def mse_with_variance_penalty(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32)
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            pred_std = tf.math.reduce_std(y_pred)
            variance_penalty = tf.square(tf.maximum(0.0, target_std_constant / 2.0 - pred_std))
            return mse_loss + 0.1 * variance_penalty

        self.model.compile(
            optimizer=optimizer,
            loss=mse_with_variance_penalty,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None and y_val is not None else "loss",
                patience=30,
                restore_best_weights=True,
                min_delta=1e-5,
                mode="min",
                verbose=1,
            )
        ]
        if model_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_path,
                    monitor="val_loss" if X_val is not None and y_val is not None else "loss",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="min",
                    verbose=0,
                )
            )

        fit_kwargs: Dict[str, Any] = {
            "x": X_train_scaled,
            "y": y_train_scaled,
            "epochs": epochs,
            "batch_size": batch_size,
            "shuffle": True,
            "callbacks": callbacks,
            "verbose": 0,
        }

        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (
                self.transform_features(X_val),
                self.transform_target(y_val),
            )

        history = self.model.fit(**fit_kwargs)
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.transform_features(X)
        prediction = self.model.predict(X_scaled, verbose=0).reshape(-1)
        return self.inverse_transform_target(prediction)

    def extract_encoder_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.transform_features(X)
        feature_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("final_layer_norm").output,
        )
        encoded = feature_model.predict(X_scaled, verbose=0)
        return encoded[:, -1, :]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_scaled = self.transform_features(X)
        y_scaled = self.transform_target(y)
        results = self.model.evaluate(X_scaled, y_scaled, verbose=0, return_dict=True)
        return {key: float(value) for key, value in results.items()}


def create_transformer_model(**kwargs) -> TransformerTimeSeriesModel:
    return TransformerTimeSeriesModel(**kwargs)
