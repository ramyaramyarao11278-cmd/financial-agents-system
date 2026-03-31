from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from .base_agent import BaseAgent
from utils.indicators import (
    calculate_bollinger_bands, calculate_ema,
    calculate_macd, calculate_rsi, calculate_sma,
    calculate_stochastics,
    calculate_atr, calculate_cci,
    calculate_momentum, calculate_obv,
    calculate_mfi
)
from utils.logger import get_logger
from utils.transformer_model import create_transformer_model
from utils.classifier_model import create_classifier_model

logger = get_logger(__name__)

class TechnicalAnalystAgent(BaseAgent):
    """Technical Analyst Agent responsible for fusing sentiment analysis with technical indicators
    using Transformer model with attention mechanism for time series prediction."""
    
    def __init__(self):
        super().__init__(
            name="Technical Analyst",
            description="Fuses sentiment analysis with technical indicators using Transformer model with attention mechanism for time series prediction"
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            requested_indicators = input_data.get(
                "indicators",
                ["sma", "ema", "rsi", "macd", "bollinger", "stochastics"]
            )
            prediction_steps = input_data.get("prediction_steps", 7)
            sentiment_results = input_data.get("sentiment_results") or {}

            data_by_symbol = input_data.get("data_by_symbol")
            if not data_by_symbol:
                if "data" not in input_data:
                    return self._format_output(
                        status="error",
                        result=None,
                        message="Missing 'data_by_symbol' or 'data'"
                    )
                data_by_symbol = self._split_price_by_symbol(input_data["data"])

            logger.info(f"Performing multi-symbol technical analysis: {list(data_by_symbol.keys())}")

            results_by_symbol = {}

            for symbol, records in data_by_symbol.items():
                df = pd.DataFrame(records)
                if df.empty:
                    continue

                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").set_index("Date")

                if "Symbol" in df.columns:
                    df = df.drop(columns=["Symbol"])

                indicator_results = self._calculate_indicators(df, requested_indicators)
                signals = self._generate_signals(df, indicator_results)
                trend_analysis = self._analyze_trend(df)
                enhanced_df = self._create_enhanced_dataframe(df, indicator_results)

                symbol_sentiment = sentiment_results.get("by_symbol", {}).get(symbol, {})
                sentiment_df = self._prepare_sentiment_data(symbol_sentiment, enhanced_df) if symbol_sentiment else None

                predictions = self._predict_time_series(
                    symbol=symbol,
                    enhanced_df=enhanced_df,
                    sentiment_df=sentiment_df,
                    prediction_steps=prediction_steps
                )

                technical_indicators_data = enhanced_df.reset_index().copy()
                technical_indicators_data["Date"] = pd.to_datetime(technical_indicators_data["Date"]).dt.strftime("%Y-%m-%d")

                results_by_symbol[symbol] = {
                    "symbol": symbol,
                    "indicators": indicator_results,
                    "signals": signals,
                    "trend_analysis": trend_analysis,
                    "predictions": predictions,
                    "technical_indicators_data": technical_indicators_data.to_dict("records"),
                    "model_type": "transformer_attention"
                }

            result = {
                "results_by_symbol": results_by_symbol,
                "model_type": "transformer_attention"
            }

            return self._format_output(
                status="success",
                result=result,
                message="Successfully performed technical analysis and prediction for CSI100 and CSI300"
            )

        except Exception as e:
            logger.error(f"Error in TechnicalAnalystAgent: {str(e)}")
            return self._format_output(
                status="error",
                result=None,
                message=str(e)
            )
    def _split_price_by_symbol(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        results = {}

        for row in data:
            symbol = row.get("Symbol", "UNKNOWN")
            results.setdefault(symbol, []).append(row)

        return results
    
    def _calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """
        Calculate the requested technical indicators.
        
        Args:
            df: DataFrame containing price data.
            indicators: List of indicators to calculate.
        
        Returns:
            Dictionary containing calculated indicators.
        """
        results = {}
        
        # Extract close prices
        close_prices = df["Close"]
        
        if "sma" in indicators:
            results["sma"] = {
                "sma_20": calculate_sma(close_prices, 20).tolist(),
                "sma_50": calculate_sma(close_prices, 50).tolist(),
                "sma_200": calculate_sma(close_prices, 200).tolist()
            }
        
        if "ema" in indicators:
            results["ema"] = {
                "ema_12": calculate_ema(close_prices, 12).tolist(),
                "ema_26": calculate_ema(close_prices, 26).tolist()
            }
        
        if "rsi" in indicators:
            results["rsi"] = calculate_rsi(close_prices).tolist()
        
        if "macd" in indicators:
            macd_line, signal_line, histogram = calculate_macd(close_prices)
            results["macd"] = {
                "macd_line": macd_line.tolist(),
                "signal_line": signal_line.tolist(),
                "histogram": histogram.tolist()
            }
        
        if "bollinger" in indicators:
            upper, middle, lower = calculate_bollinger_bands(close_prices)
            results["bollinger"] = {
                "upper": upper.tolist(),
                "middle": middle.tolist(),
                "lower": lower.tolist()
            }
        
        if "stochastics" in indicators:
            k, d = calculate_stochastics(df)
            results["stochastics"] = {
                "k": k.tolist(),
                "d": d.tolist()
            }
            # ======================
        # ATR (Volatility)
        # ======================
        results["atr"] = calculate_atr(df).tolist()

        # ======================
        # CCI (Trend strength)
        # ======================
        results["cci"] = calculate_cci(df).tolist()

        # ======================
        # Momentum
        # ======================
        results["momentum"] = calculate_momentum(close_prices).tolist()

        # ======================
        # OBV (Volume trend)
        # ======================
        if "Volume" in df.columns:
            results["obv"] = calculate_obv(df).tolist()

        # ======================
        # MFI (Money flow)
        # ======================
        if "Volume" in df.columns:
            results["mfi"] = calculate_mfi(df).tolist()
        return results
    
    def _generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame containing price data.
            indicators: Dictionary containing calculated indicators.
        
        Returns:
            Dictionary containing trading signals.
        """
        signals = []
        
        # Simple signal generation logic (to be expanded)
        if "rsi" in indicators:
            rsi_values = indicators["rsi"]
            for i, rsi in enumerate(rsi_values):
                if rsi < 30:
                    signals.append({
                        "date": str(df.index[i]),
                        "signal": "buy",
                        "indicator": "rsi",
                        "value": rsi,
                        "reason": "RSI below 30 (oversold)"
                    })
                elif rsi > 70:
                    signals.append({
                        "date": str(df.index[i]),
                        "signal": "sell",
                        "indicator": "rsi",
                        "value": rsi,
                        "reason": "RSI above 70 (overbought)"
                    })
        
        return signals
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the overall trend of the stock.
        
        Args:
            df: DataFrame containing price data.
        
        Returns:
            Dictionary containing trend analysis.
        """
        close_prices = df["Close"]
        
        # Calculate short-term and long-term SMAs
        sma_20 = calculate_sma(close_prices, 20)
        sma_200 = calculate_sma(close_prices, 200)
        
        # Determine trend direction
        if sma_20.iloc[-1] > sma_200.iloc[-1] and sma_20.iloc[-1] > sma_20.iloc[-20]:
            trend = "bullish"
        elif sma_20.iloc[-1] < sma_200.iloc[-1] and sma_20.iloc[-1] < sma_20.iloc[-20]:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "sma_20": sma_20.iloc[-1],
            "sma_200": sma_200.iloc[-1],
            "price_change_30d": ((close_prices.iloc[-1] - close_prices.iloc[-30]) / close_prices.iloc[-30] * 100).round(2)
        }
    
    def _create_enhanced_dataframe(self, df: pd.DataFrame, indicator_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create an enhanced dataframe with technical indicators.
        
        Args:
            df: Original price data dataframe
            indicator_results: Calculated indicators
            
        Returns:
            Enhanced dataframe with indicators added as columns
        """
        logger.info("Creating enhanced dataframe with technical indicators")
        
        # Create a copy of the original dataframe
        enhanced_df = df.copy()
        
        # Add indicators to the dataframe
        if "sma" in indicator_results:
            enhanced_df["sma_20"] = indicator_results["sma"]["sma_20"]
            # Don't add longer SMAs that require more data
            # enhanced_df["sma_50"] = indicator_results["sma"]["sma_50"]
            # enhanced_df["sma_200"] = indicator_results["sma"]["sma_200"]
        
        if "ema" in indicator_results:
            enhanced_df["ema_12"] = indicator_results["ema"]["ema_12"]
            enhanced_df["ema_26"] = indicator_results["ema"]["ema_26"]
        
        if "rsi" in indicator_results:
            enhanced_df["rsi"] = indicator_results["rsi"]
        
        if "macd" in indicator_results:
            enhanced_df["macd_line"] = indicator_results["macd"]["macd_line"]
            enhanced_df["signal_line"] = indicator_results["macd"]["signal_line"]
            enhanced_df["macd_histogram"] = indicator_results["macd"]["histogram"]
        
        if "bollinger" in indicator_results:
            enhanced_df["bollinger_upper"] = indicator_results["bollinger"]["upper"]
            enhanced_df["bollinger_middle"] = indicator_results["bollinger"]["middle"]
            enhanced_df["bollinger_lower"] = indicator_results["bollinger"]["lower"]
        
        if "stochastics" in indicator_results:
            enhanced_df["stochastics_k"] = indicator_results["stochastics"]["k"]
            enhanced_df["stochastics_d"] = indicator_results["stochastics"]["d"]
        
        if "atr" in indicator_results:
            enhanced_df["atr"] = indicator_results["atr"]

        if "cci" in indicator_results:
            enhanced_df["cci"] = indicator_results["cci"]

        if "momentum" in indicator_results:
            enhanced_df["momentum"] = indicator_results["momentum"]

        if "obv" in indicator_results:
            enhanced_df["obv"] = indicator_results["obv"]

        if "mfi" in indicator_results:
            enhanced_df["mfi"] = indicator_results["mfi"]

        # 仅保留经筛选的少量新增特征，避免维度膨胀淹没有效信号。
        daily_return = enhanced_df["Close"].pct_change()

        for lag in range(1, 4):
            enhanced_df[f"lag_return_{lag}"] = daily_return.shift(lag)

        enhanced_df["realized_vol_5"] = daily_return.rolling(window=5).std()
        enhanced_df["momentum_5"] = enhanced_df["Close"].pct_change(periods=5)
        
        # Drop rows with missing values (due to indicators needing warm-up period)
        enhanced_df = enhanced_df.dropna()
        
        return enhanced_df
    
    def _prepare_sentiment_data(self, sentiment_results: Dict[str, Any], enhanced_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preparing precomputed sentiment data for date alignment")

        raw_dimensions = sentiment_results.get("sentiment_dimensions")
        if not raw_dimensions:
            return None

        if isinstance(raw_dimensions, dict):
            sent_df = pd.DataFrame(list(raw_dimensions.values()))
        else:
            sent_df = pd.DataFrame(raw_dimensions)

        if sent_df.empty:
            return None

        sent_df["trade_date"] = pd.to_datetime(sent_df["trade_date"]).dt.normalize()
        sent_df = sent_df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

        if "Symbol" in sent_df.columns:
            sent_df = sent_df.drop(columns=["Symbol"])

        numeric_cols = sent_df.select_dtypes(include=["number"]).columns.tolist()
        sent_df = sent_df[["trade_date"] + numeric_cols].set_index("trade_date")

        sent_df = sent_df.shift(1)
        sent_df = self._add_sentiment_derived_features(sent_df)
        aligned = sent_df.reindex(pd.to_datetime(enhanced_df.index).normalize())
        aligned.index = enhanced_df.index
        aligned = aligned.ffill(limit=1)

        valid_mask = ~aligned.isna().all(axis=1)
        aligned = aligned.loc[valid_mask]

        return aligned

    def _add_sentiment_derived_features(self, sent_df: pd.DataFrame) -> pd.DataFrame:
        if sent_df.empty:
            return sent_df

        sent_df = sent_df.copy()

        score_col = None
        for candidate in ["combined_sentiment", "avg_tone", "weighted_tone", "quality_tone"]:
            if candidate in sent_df.columns:
                score_col = candidate
                break

        # 基于主情感分数构造平滑、变化率和波动特征。
        if score_col is not None:
            sent_df["sentiment_ma3"] = sent_df[score_col].rolling(window=3).mean()
            sent_df["sentiment_change"] = sent_df[score_col].diff()

        return sent_df
    def _build_combined_dataframe(self, enhanced_df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        combined_df = enhanced_df.copy()

        if sentiment_df is not None and not sentiment_df.empty:
            common_index = combined_df.index.intersection(sentiment_df.index)
            combined_df = combined_df.loc[common_index].copy()
            sentiment_use = sentiment_df.loc[common_index].copy()

            common_cols = set(combined_df.columns) & set(sentiment_use.columns)
            if common_cols:
                sentiment_use = sentiment_use.rename(columns={col: f"sentiment_{col}" for col in common_cols})

            combined_df = combined_df.join(sentiment_use, how="inner")

        combined_df = combined_df.sort_index().dropna()
        combined_df = self._select_experiment_features(combined_df)
        logger.info("Final input feature dimension: %s", len(combined_df.columns))
        logger.info("Final input feature columns: %s", list(combined_df.columns))
        return combined_df

    def _select_experiment_features(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        base_candidates = ["Close", "Volume"]
        if "Turnover" in combined_df.columns:
            turnover_corr = combined_df[["Volume", "Turnover"]].corr().iloc[0, 1]
            if pd.isna(turnover_corr) or abs(float(turnover_corr)) <= 0.95:
                base_candidates.append("Turnover")
        feature_candidates = [
            "rsi",
            "stochastics_k",
            "macd_histogram",
            "obv",
            "momentum_5",
            "realized_vol_5",
            "lag_return_1",
            "lag_return_2",
            "lag_return_3",
        ]

        selected_columns = [col for col in base_candidates if col in combined_df.columns]
        selected_columns.extend([col for col in feature_candidates if col in combined_df.columns])
        selected_columns.extend([col for col in combined_df.columns if col.startswith("sentiment_") or col.startswith("news_")])

        selected_columns = list(dict.fromkeys(selected_columns))
        selected_df = combined_df[selected_columns].copy()

        corr_df = selected_df.corr().abs()
        dropped_columns = []
        for row_idx in range(len(corr_df.columns)):
            for col_idx in range(row_idx + 1, len(corr_df.columns)):
                left_col = corr_df.index[row_idx]
                right_col = corr_df.columns[col_idx]
                corr_value = float(corr_df.iloc[row_idx, col_idx])
                if corr_value > 0.85 and right_col in selected_df.columns:
                    selected_df = selected_df.drop(columns=[right_col])
                    dropped_columns.append((left_col, right_col, corr_value))

        corr_df = selected_df.corr().abs()
        high_corr_pairs = []
        for row_idx in range(len(corr_df.columns)):
            for col_idx in range(row_idx + 1, len(corr_df.columns)):
                corr_value = float(corr_df.iloc[row_idx, col_idx])
                if corr_value > 0.85:
                    high_corr_pairs.append((corr_df.index[row_idx], corr_df.columns[col_idx], corr_value))

        high_corr_pairs = sorted(high_corr_pairs, key=lambda item: item[2], reverse=True)
        logger.info("Post-pruning feature dimension: %s", len(selected_df.columns))
        logger.info("Post-pruning feature columns: %s", list(selected_df.columns))
        logger.info(
            "Dropped highly correlated columns (threshold 0.85): %s",
            [(left, right, round(corr_value, 4)) for left, right, corr_value in dropped_columns] if dropped_columns else "None",
        )
        if high_corr_pairs:
            logger.info(
                "Post-pruning high-correlation feature pairs (|corr| > 0.85): %s",
                [(left, right, round(corr_value, 4)) for left, right, corr_value in high_corr_pairs],
            )
        else:
            logger.info("Post-pruning high-correlation feature pairs (|corr| > 0.85): None")

        return selected_df

    def _get_target_horizon(self, symbol: str) -> int:
        return 3 if symbol == "CSI300" else 1

    def _build_prediction_targets(self, combined_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        horizon = self._get_target_horizon(symbol)
        target_df = combined_df.copy()
        daily_return = target_df["Close"].pct_change()
        future_return = sum(daily_return.shift(-step) for step in range(1, horizon + 1))
        target_df["target_return"] = future_return
        target_df["target_direction"] = (future_return > 0).astype(int)
        target_df = target_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_return", "target_direction"])
        return target_df

    def _create_window_dataset(
        self,
        target_df: pd.DataFrame,
        sequence_length: int,
    ):
        feature_columns = target_df.select_dtypes(include=["number"]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in {"target_return", "target_direction"}]

        X_windows = []
        y_return = []
        y_direction = []
        sample_dates = []
        anchor_close = []

        for end_idx in range(sequence_length - 1, len(target_df)):
            window_df = target_df.iloc[end_idx - sequence_length + 1 : end_idx + 1]
            X_windows.append(window_df[feature_columns].to_numpy(dtype=np.float32))
            y_return.append(float(target_df["target_return"].iloc[end_idx]))
            y_direction.append(int(target_df["target_direction"].iloc[end_idx]))
            sample_dates.append(target_df.index[end_idx])
            anchor_close.append(float(target_df["Close"].iloc[end_idx]))

        return (
            np.asarray(X_windows, dtype=np.float32),
            np.asarray(y_return, dtype=np.float32),
            np.asarray(y_direction, dtype=np.int32),
            sample_dates,
            np.asarray(anchor_close, dtype=np.float32),
            feature_columns,
        )
    
    def _predict_time_series(self, symbol: str, enhanced_df: pd.DataFrame, sentiment_df: pd.DataFrame, prediction_steps: int) -> Dict[str, Any]:
        logger.info(f"Performing time series prediction for {symbol}")

        combined_df = self._build_combined_dataframe(enhanced_df, sentiment_df)

        if combined_df.empty or len(combined_df) < 80:
            return {
                "symbol": symbol,
                "prediction_steps": prediction_steps,
                "future_prices": [],
                "prediction_dates": [],
                "test_dates": [],
                "test_actual_prices": [],
                "test_predicted_prices": [],
                "model_evaluation": None,
                "sequence_length": 30,
                "model_name": "transformer_attention",
                "message": "Insufficient aligned data for prediction"
            }

        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            from utils.evaluation import find_best_threshold, regression_metrics
            from utils.prediction_exporter import (
                export_confusion_matrix_plot,
                export_prediction_csv,
                export_price_plot,
                export_return_plot,
                export_roc_plot,
            )

            sequence_length = 20
            target_df = self._build_prediction_targets(combined_df, symbol)
            X, y_return, y_direction, sample_dates, anchor_close, feature_columns = self._create_window_dataset(
                target_df,
                sequence_length=sequence_length,
            )

            if len(X) < 10:
                return {
                    "symbol": symbol,
                    "prediction_steps": prediction_steps,
                    "future_prices": [],
                    "prediction_dates": [],
                    "test_dates": [],
                    "test_actual_prices": [],
                    "test_predicted_prices": [],
                    "model_evaluation": None,
                    "sequence_length": sequence_length,
                    "model_name": "transformer_attention",
                    "message": "Insufficient samples after sequence preparation"
                }

            split_idx = max(1, min(len(X) - 1, int(len(X) * 0.8)))

            X_train_all, X_test = X[:split_idx], X[split_idx:]
            y_return_train_all, y_return_test = y_return[:split_idx], y_return[split_idx:]
            y_direction_train_all, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
            test_dates = sample_dates[split_idx:]
            test_anchor_close = anchor_close[split_idx:]

            if len(X_train_all) >= 20:
                val_size = max(1, int(len(X_train_all) * 0.2))
                train_end = len(X_train_all) - val_size
                X_train, X_val = X_train_all[:train_end], X_train_all[train_end:]
                y_return_train, y_return_val = y_return_train_all[:train_end], y_return_train_all[train_end:]
                y_direction_train, y_direction_val = y_direction_train_all[:train_end], y_direction_train_all[train_end:]
            else:
                X_train, y_return_train, y_direction_train = X_train_all, y_return_train_all, y_direction_train_all
                X_val = y_return_val = y_direction_val = None

            batch_size = min(32, len(X_train)) if len(X_train) > 0 else 1
            num_features = X.shape[-1]

            regression_model = create_transformer_model(
                sequence_length=sequence_length,
                num_features=num_features,
                d_model=64,
                num_heads=4,
                dff=128,
                num_transformer_layers=2,
                dropout_rate=0.2,
            )
            classifier_model = create_classifier_model(
                sequence_length=sequence_length,
                num_features=num_features,
                d_model=32,
                num_heads=4,
                dff=64,
                dropout_rate=0.3,
            )

            regression_model_path = f"outputs/{symbol}_regression_model.weights.h5"
            classifier_model_path = f"outputs/{symbol}_classifier_model.weights.h5"

            regression_model.train(
                X_train,
                y_return_train,
                X_val=X_val,
                y_val=y_return_val,
                epochs=300,
                batch_size=batch_size,
                model_path=regression_model_path,
            )
            classifier_model.train(
                X_train,
                y_direction_train,
                X_val=X_val,
                y_val=y_direction_val,
                epochs=300,
                batch_size=batch_size,
                model_path=classifier_model_path,
            )

            if symbol == "CSI300":
                logger.info(
                    "%s samples | train=%s val=%s test=%s | positive_ratio train=%.4f val=%.4f test=%.4f",
                    symbol,
                    len(X_train),
                    len(X_val) if X_val is not None else 0,
                    len(X_test),
                    float(np.mean(y_direction_train)),
                    float(np.mean(y_direction_val)) if y_direction_val is not None else 0.0,
                    float(np.mean(y_direction_test)),
                )

            evaluation = regression_model.evaluate(X_test, y_return_test) if len(X_test) > 0 else {}

            reg_metrics = {}
            cls_metrics = {}
            csv_file = None
            metrics_file = None
            return_plot_file = None
            price_plot_file = None
            confusion_matrix_file = None
            roc_plot_file = None
            test_dates_str = []
            actual_prices = []
            predicted_prices = []
            actual_returns = []
            predicted_returns = []

            if len(X_test) > 0:
                predicted_returns = regression_model.predict(X_test).tolist()
                actual_returns = y_return_test.tolist()
                test_dates_str = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in test_dates]

                reg_metrics = regression_metrics(actual_returns, predicted_returns)

                true_trend = y_direction_test.astype(int)
                direction_prob = np.asarray(classifier_model.predict_proba(X_test), dtype=float)
                probability_inverted = False
                if len(np.unique(true_trend)) > 1 and roc_auc_score(true_trend, direction_prob) < 0.5:
                    direction_prob = 1.0 - direction_prob
                    probability_inverted = True

                if X_val is not None and y_direction_val is not None:
                    val_prob = np.asarray(classifier_model.predict_proba(X_val), dtype=float)
                    val_true = y_direction_val.astype(int)
                    if len(np.unique(val_true)) > 1 and roc_auc_score(val_true, val_prob) < 0.5:
                        val_prob = 1.0 - val_prob

                    best_threshold, cls_metrics, top_candidates = find_best_threshold(val_true, val_prob)
                    logger.info("Validation threshold candidates for %s: %s", symbol, top_candidates)
                    logger.info("Selected validation threshold for %s: %.3f | metrics=%s", symbol, best_threshold, cls_metrics)
                else:
                    best_threshold, cls_metrics, top_candidates = find_best_threshold(true_trend, direction_prob)

                pred_trend = (direction_prob >= best_threshold).astype(int)

                base_close = float(test_anchor_close[0]) if len(test_anchor_close) > 0 else float(combined_df["Close"].iloc[-1])
                actual_prices = ((1 + pd.Series(actual_returns)).cumprod() * base_close).tolist()
                predicted_prices = ((1 + pd.Series(predicted_returns)).cumprod() * base_close).tolist()

                csv_file, metrics_file = export_prediction_csv(
                    symbol=symbol,
                    dates=test_dates_str,
                    actual_returns=actual_returns,
                    predicted_returns=predicted_returns,
                    actual_prices=actual_prices,
                    predicted_prices=predicted_prices,
                    direction_prob=direction_prob.tolist(),
                    direction_label=pred_trend.tolist(),
                    output_dir="outputs",
                    classification_threshold=best_threshold,
                )

                return_plot_file = export_return_plot(
                    symbol=symbol,
                    dates=test_dates_str,
                    actual=actual_returns,
                    predicted=predicted_returns,
                    output_dir="outputs"
                )

                price_plot_file = export_price_plot(
                    symbol=symbol,
                    dates=test_dates_str,
                    actual=actual_prices,
                    predicted=predicted_prices,
                    output_dir="outputs"
                )

                confusion_matrix_file = export_confusion_matrix_plot(
                    symbol=symbol,
                    y_true=true_trend,
                    y_pred=pred_trend,
                    output_dir="outputs"
                )

                if len(np.unique(true_trend)) > 1:
                    fpr, tpr, _ = roc_curve(true_trend, direction_prob)
                    roc_plot_file = export_roc_plot(
                        symbol=symbol,
                        fpr=fpr,
                        tpr=tpr,
                        auc_score=cls_metrics.get("AUC_ROC", 0.0),
                        output_dir="outputs"
                    )

                cls_metrics["Best_Threshold"] = float(best_threshold)
                cls_metrics["Probability_Inverted"] = probability_inverted
                cls_metrics["Regression_Model"] = "transformer_regression"
                cls_metrics["Classifier_Model"] = "transformer_classifier"

            future_predictions = []

            last_date = pd.to_datetime(combined_df.index[-1])
            future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=prediction_steps)
            future_dates_str = [d.strftime("%Y-%m-%d") for d in future_dates]

            combined_records = combined_df.reset_index().copy()
            combined_records["Date"] = pd.to_datetime(combined_records["Date"]).dt.strftime("%Y-%m-%d")

            return {
                "symbol": symbol,
                "prediction_steps": prediction_steps,
                "future_prices": future_predictions,
                "prediction_dates": future_dates_str,
                "test_dates": test_dates_str,
                "test_actual_prices": actual_prices,
                "test_predicted_prices": predicted_prices,
                "test_actual_returns": actual_returns,
                "test_predicted_returns": predicted_returns,
                "model_evaluation": {
                    "transformer": evaluation,
                    "regression": reg_metrics,
                    "trend": cls_metrics
                },
                "sequence_length": sequence_length,
                "model_name": "transformer_attention",
                "result_file": csv_file,
                "metrics_file": metrics_file,
                "return_curve_file": return_plot_file,
                "price_curve_file": price_plot_file,
                "confusion_matrix_file": confusion_matrix_file,
                "roc_curve_file": roc_plot_file,
                "transformer_input_data": combined_records.to_dict("records")
            }

        except Exception as e:
            logger.error(f"Error during prediction for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "symbol": symbol,
                "prediction_steps": prediction_steps,
                "future_prices": [],
                "prediction_dates": [],
                "test_dates": [],
                "test_actual_prices": [],
                "test_predicted_prices": [],
                "model_evaluation": None,
                "sequence_length": 30,
                "model_name": "transformer_attention",
                "message": f"Prediction error: {str(e)}"
            }
