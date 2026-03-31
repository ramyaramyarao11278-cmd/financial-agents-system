#!/usr/bin/env python3
"""
Trading environment implementation for reinforcement learning.
"""

from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete

from utils.logger import get_logger

logger = get_logger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning with financial time series data.

    Action Space:
        - 0: Hold
        - 1: Buy
        - 2: Sell
    """

    metadata = {
        "render.modes": ["human", "none"]
    }

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000, transaction_cost: float = 0.001):
        super().__init__()

        self.window_size = 30
        self.data = self._prepare_env_dataframe(data)

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.action_space = Discrete(3)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(self.window_size * len(self.data.columns) + 4,),
            dtype=np.float32
        )

        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0
        self.total_portfolio_value = initial_capital
        self.position = 0
        self.net_worth_history = []

        self.normalized_data = self._normalize_data()

    def _prepare_env_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.copy()

        if isinstance(prepared.index, pd.DatetimeIndex):
            pass
        elif not pd.api.types.is_numeric_dtype(prepared.index):
            logger.info("TradingEnv: Resetting index to make it numeric")
            prepared = prepared.reset_index(drop=True)

        core_price_cols = ["Open", "High", "Low", "Close", "Volume"]

        for col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="ignore")

        for col in core_price_cols:
            if col in prepared.columns:
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

        numeric_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()

        keep_cols = []
        for col in core_price_cols + numeric_cols:
            if col in prepared.columns and col not in keep_cols:
                keep_cols.append(col)

        prepared = prepared[keep_cols].copy()
        prepared = prepared.loc[:, ~prepared.columns.duplicated()]

        if "Close" not in prepared.columns:
            raise KeyError(
                f"TradingEnv requires a 'Close' column. Available columns: {list(prepared.columns)}"
            )

        prepared = prepared.dropna(subset=["Close"])

        for col in prepared.columns:
            if col in core_price_cols:
                prepared[col] = prepared[col].ffill()
            else:
                prepared[col] = prepared[col].fillna(0)

        return prepared

    def _normalize_data(self) -> pd.DataFrame:
        normalized = self.data.copy()

        for col in normalized.columns:
            try:
                if not pd.api.types.is_numeric_dtype(normalized[col]):
                    logger.warning(f"Column {col} is not numeric, skipping normalization")
                    normalized[col] = 0.0
                    continue

                min_val = normalized[col].min()
                max_val = normalized[col].max()

                if max_val != min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
                else:
                    normalized[col] = 0.0
            except Exception as e:
                logger.warning(f"Failed to normalize column {col}: {str(e)}")
                normalized[col] = 0.0

        return normalized

    def _safe_close(self, step: int) -> float:
        if "Close" not in self.data.columns:
            raise KeyError(
                f"'Close' column not found in trading environment data. "
                f"Available columns: {list(self.data.columns)}"
            )

        try:
            return float(self.data.iloc[step]["Close"])
        except Exception as e:
            logger.warning(f"Failed to get Close price at step {step}: {e}")
            return 0.0

    def _get_observation(self) -> np.ndarray:
        start = max(0, self.current_step - self.window_size)
        window = self.normalized_data.iloc[start:self.current_step]

        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), window.shape[1]))
            window_values = np.vstack([padding, window.values])
        else:
            window_values = window.values

        cash_norm = self.cash / (self.initial_capital * 10)
        close_price = self._safe_close(self.current_step)
        holdings_norm = (self.holdings * close_price) / (self.initial_capital * 10)
        portfolio_norm = self.total_portfolio_value / (self.initial_capital * 10)

        try:
            obs = np.concatenate([
                window_values.flatten(),
                np.array([cash_norm, holdings_norm, portfolio_norm], dtype=np.float32),
                np.array([float(self.position)], dtype=np.float32)
            ])
        except Exception as e:
            logger.error(f"Failed to create observation array: {e}")
            obs = np.zeros(window_values.size + 4, dtype=np.float32)

        return obs

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> tuple:
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.holdings = 0
        self.total_portfolio_value = self.initial_capital
        self.position = 0
        self.net_worth_history = [self.initial_capital]

        return self._get_observation(), {}

    def step(self, action: int) -> tuple:
        current_price = self._safe_close(self.current_step)
        previous_value = self.total_portfolio_value

        if action == 1:
            if self.cash > 0 and self.position == 0 and current_price > 0:
                shares_to_buy = (self.cash * 0.2) // current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee

                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.holdings += shares_to_buy
                        self.position = 1

        elif action == 2:
            if self.holdings > 0 and self.position == 1 and current_price > 0:
                proceeds = self.holdings * current_price
                transaction_fee = proceeds * self.transaction_cost
                net_proceeds = proceeds - transaction_fee

                self.cash += net_proceeds
                self.holdings = 0
                self.position = 0

        holdings_value = self.holdings * current_price
        self.total_portfolio_value = self.cash + holdings_value
        self.net_worth_history.append(self.total_portfolio_value)

        reward = 0.0
        if previous_value > 0:
            reward = (self.total_portfolio_value - previous_value) / previous_value

        prev_close = self._safe_close(max(self.current_step - 1, 0))
        if prev_close > 0:
            price_change = (current_price - prev_close) / prev_close
            reward += price_change * self.position

        if action != 0:
            reward -= 0.001

        if self.total_portfolio_value > previous_value:
            reward += 0.001
        elif self.total_portfolio_value < previous_value:
            reward -= 0.001

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        if not done:
            observation = self._get_observation()
        else:
            obs_shape = (self.window_size * len(self.normalized_data.columns) + 4,)
            observation = np.zeros(obs_shape, dtype=np.float32)

        info = {
            "step": self.current_step,
            "cash": self.cash,
            "holdings": self.holdings,
            "holdings_value": holdings_value,
            "total_portfolio_value": self.total_portfolio_value,
            "current_price": current_price,
            "position": self.position,
            "action": action
        }

        return observation, reward, done, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            logger.info(f"Step: {self.current_step}")
            logger.info(f"Cash: ${self.cash:.2f}")
            logger.info(f"Holdings: {self.holdings} shares")

            current_price = self._safe_close(min(self.current_step, len(self.data) - 1))
            logger.info(f"Holdings Value: ${self.holdings * current_price:.2f}")
            logger.info(f"Total Portfolio Value: ${self.total_portfolio_value:.2f}")
            logger.info(f"Position: {'Long' if self.position == 1 else 'None'}")
            logger.info("-" * 50)

    def get_portfolio_stats(self) -> Dict[str, Any]:
        if not self.net_worth_history:
            return {}

        returns = np.diff(self.net_worth_history) / np.array(self.net_worth_history[:-1])

        total_return = (self.total_portfolio_value - self.initial_capital) / self.initial_capital * 100
        annualized_return = (1 + total_return / 100) ** (252 / len(self.data)) - 1
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0

        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / (volatility / 100) if volatility != 0 else 0

        rolling_max = np.maximum.accumulate(self.net_worth_history)
        drawdown = (np.array(self.net_worth_history) - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": self.total_portfolio_value,
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return * 100, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "net_worth_history": self.net_worth_history
        }

    def close(self) -> None:
        pass