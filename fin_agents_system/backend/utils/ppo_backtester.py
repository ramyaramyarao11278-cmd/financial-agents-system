#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) backtester implementation for financial time series data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.logger import get_logger
from utils.trading_env import TradingEnv

logger = get_logger(__name__)


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class PPOBacktester:
    """
    PPO-based backtesting system for trading strategies.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_range: str = "1y",
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        sentiment_data: pd.DataFrame = None,
        technical_results: Dict[str, Any] = None
    ):
        self.data = data.copy()

        range_map = {
            "7d": 7,
            "1m": 30,
            "3m": 90,
            "1y": 365,
            "2y": 730
        }

        if time_range in range_map:
            max_days = range_map[time_range]
            if len(self.data) > max_days:
                self.data = self.data.tail(max_days)

        logger.info(f"PPOBacktester: Using last {len(self.data)} days for training")

        # 先标准化基础行情数据，必须保住 OHLCV
        self.data = self._prepare_env_dataframe(self.data, context="PPOBacktester base data: ")
        logger.info(f"PPOBacktester: Base market data shape after preprocessing: {self.data.shape}")

        # Merge sentiment data if provided
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_use = sentiment_data.copy()

            if isinstance(sentiment_use.index, pd.DatetimeIndex):
                aligned_sentiment = sentiment_use.reindex(self.data.index)
            else:
                if "Date" in sentiment_use.columns:
                    sentiment_use["Date"] = pd.to_datetime(sentiment_use["Date"])
                    sentiment_use = sentiment_use.set_index("Date")
                aligned_sentiment = sentiment_use.reindex(self.data.index)

            aligned_sentiment = aligned_sentiment.ffill(limit=1)

            # 情绪特征只保留数值列
            for col in aligned_sentiment.columns:
                aligned_sentiment[col] = pd.to_numeric(aligned_sentiment[col], errors="coerce")

            aligned_sentiment = aligned_sentiment.dropna(axis=1, how="all").fillna(0)

            if not aligned_sentiment.empty:
                self.data = pd.concat([self.data, aligned_sentiment], axis=1)
                logger.info(f"PPOBacktester: Merged sentiment data with {len(aligned_sentiment.columns)} columns")

        # Merge technical indicators if provided
        if technical_results is not None:
            technical_indicators = technical_results.get("technical_indicators_df")

            if technical_indicators is None:
                technical_indicators = technical_results.get("technical_indicators_data")

            if isinstance(technical_indicators, list):
                technical_indicators = pd.DataFrame(technical_indicators)

            if technical_indicators is not None and not isinstance(technical_indicators, pd.DataFrame):
                technical_indicators = pd.DataFrame(technical_indicators)

            if technical_indicators is not None and not technical_indicators.empty:
                technical_use = technical_indicators.copy()

                if "Date" in technical_use.columns:
                    technical_use["Date"] = pd.to_datetime(technical_use["Date"])
                    technical_use = technical_use.set_index("Date")

                if "Symbol" in technical_use.columns:
                    technical_use = technical_use.drop(columns=["Symbol"])

                technical_use = technical_use.reindex(self.data.index)

                for col in technical_use.columns:
                    technical_use[col] = pd.to_numeric(technical_use[col], errors="coerce")

                technical_use = technical_use.dropna(axis=1, how="all").ffill(limit=1).fillna(0)

                # 避免技术指标里重复带入 OHLCV，污染原始价格列
                duplicated_price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in technical_use.columns]
                if duplicated_price_cols:
                    logger.info(f"PPOBacktester: Dropping duplicated price columns from technical indicators: {duplicated_price_cols}")
                    technical_use = technical_use.drop(columns=duplicated_price_cols)

                if not technical_use.empty:
                    self.data = pd.concat([self.data, technical_use], axis=1)
                    logger.info(f"PPOBacktester: Merged technical indicators with {len(technical_use.columns)} columns")

        # 合并完成后再次统一清洗，确保 Close 一定存在
        self.data = self._prepare_env_dataframe(self.data, context="PPOBacktester merged data: ")
        logger.info(f"PPOBacktester: Final merged data shape: {self.data.shape}")

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.sentiment_data = sentiment_data
        self.technical_results = technical_results
        self.model = None

    def _prepare_env_dataframe(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Prepare dataframe for TradingEnv / PPO while preserving OHLCV columns.
        """
        prepared = df.copy()

        if isinstance(prepared.index, pd.DatetimeIndex):
            pass
        elif not pd.api.types.is_numeric_dtype(prepared.index):
            logger.info(f"{context}Resetting index to make it numeric")
            prepared = prepared.reset_index(drop=True)

        core_price_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing_core_cols = [col for col in core_price_cols if col in prepared.columns]

        # 先尝试强制转成数值
        for col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="ignore")

        for col in existing_core_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

        numeric_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()

        keep_cols = []
        for col in existing_core_cols + numeric_cols:
            if col not in keep_cols:
                keep_cols.append(col)

        prepared = prepared[keep_cols].copy()
        prepared = prepared.loc[:, ~prepared.columns.duplicated()]

        if "Close" not in prepared.columns:
            raise KeyError(
                f"{context}'Close' column missing after preprocessing. "
                f"Available columns: {list(prepared.columns)}"
            )

        prepared = prepared.dropna(subset=["Close"])

        for col in prepared.columns:
            if col in core_price_cols:
                prepared[col] = prepared[col].ffill()
            else:
                prepared[col] = prepared[col].fillna(0)

        return prepared

    def train(
        self,
        total_timesteps: int = 300000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        n_steps: int = 2048,
        verbose: int = 1
    ) -> Dict[str, Any]:
        import random
        import torch

        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        logger.info(f"Training PPO model for {total_timesteps} timesteps")

        def env_creator():
            preprocessed_data = self._prepare_env_dataframe(
                self.data,
                context="PPO train env: "
            )

            env = TradingEnv(
                data=preprocessed_data,
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost
            )
            return Monitor(env)

        env = DummyVecEnv([env_creator])

        # MlpPolicy 通常 CPU 更稳，也避免 SB3 的 GPU 警告
        device = "cpu"
        logger.info(f"Using device: {device} for PPO training")

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_steps=n_steps,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=verbose,
            tensorboard_log="./ppo_tensorboard/",
            device=device
        )

        callback = TrainingCallback(verbose=verbose)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )

        logger.info("PPO model training completed")

        return {
            "status": "success",
            "message": f"Model trained for {total_timesteps} timesteps",
            "model": self.model
        }

    def backtest(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not trained yet",
                "results": None
            }

        logger.info("Starting PPO backtest")

        backtest_data = test_data if test_data is not None else self.data

        preprocessed_backtest_data = self._prepare_env_dataframe(
            backtest_data,
            context="PPO backtest env: "
        )

        env = TradingEnv(
            data=preprocessed_backtest_data,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost
        )

        obs, _ = env.reset()
        done = False

        backtest_results = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "portfolio_values": [],
            "positions": [],
            "cash": [],
            "holdings": [],
            "dates": [str(d) for d in preprocessed_backtest_data.index]
        }

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(np.array(action).item())

            obs, reward, done, truncated, info = env.step(action)

            backtest_results["actions"].append(action)
            backtest_results["observations"].append(np.asarray(obs).tolist())
            backtest_results["rewards"].append(float(reward))
            backtest_results["portfolio_values"].append(float(info["total_portfolio_value"]))
            backtest_results["positions"].append(int(info["position"]))
            backtest_results["cash"].append(float(info["cash"]))
            backtest_results["holdings"].append(int(info["holdings"]))

            if truncated:
                break

        portfolio_stats = env.get_portfolio_stats()

        actions = np.array(backtest_results["actions"])
        buy_signals = int((actions == 1).sum())
        sell_signals = int((actions == 2).sum())
        hold_signals = int((actions == 0).sum())

        summary = f"""PPO Backtest Summary:
        - Initial Capital: ${self.initial_capital:,.2f}
        - Final Portfolio Value: ${portfolio_stats['final_portfolio_value']:,.2f}
        - Total Return: {portfolio_stats['total_return']:.2f}%
        - Annualized Return: {portfolio_stats['annualized_return']:.2f}%
        - Annualized Volatility: {portfolio_stats['volatility']:.2f}%
        - Sharpe Ratio: {portfolio_stats['sharpe_ratio']:.2f}
        - Maximum Drawdown: {portfolio_stats['max_drawdown']:.2f}%
        - Total Trades: {buy_signals + sell_signals}
        - Buy Signals: {buy_signals}
        - Sell Signals: {sell_signals}
        - Hold Signals: {hold_signals}
        """

        backtest_results["net_worth"] = portfolio_stats["net_worth_history"]

        results = {
            "status": "success",
            "message": "Backtest completed successfully",
            "summary": summary.strip(),
            "portfolio_stats": portfolio_stats,
            "backtest_details": backtest_results,
            "trading_metrics": {
                "total_trades": buy_signals + sell_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals
            },
            "model_info": {
                "algorithm": "PPO",
                "initial_capital": self.initial_capital,
                "transaction_cost": self.transaction_cost
            }
        }

        logger.info("PPO backtest completed")
        return results

    def save_model(self, file_path: str) -> Dict[str, Any]:
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }

        try:
            self.model.save(file_path)
            return {
                "status": "success",
                "message": f"Model saved to {file_path}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save model: {str(e)}"
            }

    def load_model(self, file_path: str) -> Dict[str, Any]:
        try:
            self.model = PPO.load(file_path)
            return {
                "status": "success",
                "message": f"Model loaded from {file_path}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }

    def generate_signals(self, data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        signal_data = data if data is not None else self.data

        preprocessed_signal_data = self._prepare_env_dataframe(
            signal_data,
            context="PPO signal env: "
        )

        env = TradingEnv(
            data=preprocessed_signal_data,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost
        )

        obs, _ = env.reset()
        done = False
        signals = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(np.array(action).item())

            obs, _, done, truncated, info = env.step(action)

            if action == 1:
                signals.append({
                    "date": str(env.data.index[env.current_step - 1]),
                    "signal": "buy",
                    "indicator": "ppo",
                    "value": float(env.data.iloc[env.current_step - 1]["Close"]),
                    "reason": "PPO model generated buy signal"
                })
            elif action == 2:
                signals.append({
                    "date": str(env.data.index[env.current_step - 1]),
                    "signal": "sell",
                    "indicator": "ppo",
                    "value": float(env.data.iloc[env.current_step - 1]["Close"]),
                    "reason": "PPO model generated sell signal"
                })

            if truncated:
                break

        return signals


def create_ppo_backtester(
    data: pd.DataFrame,
    time_range: str = "1y",
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    sentiment_data: pd.DataFrame = None,
    technical_results: Dict[str, Any] = None
) -> PPOBacktester:
    return PPOBacktester(
        data=data,
        time_range=time_range,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        sentiment_data=sentiment_data,
        technical_results=technical_results
    )