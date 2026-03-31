#!/usr/bin/env python3
"""
Backtester Agent implementation using PPO reinforcement learning.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from .base_agent import BaseAgent
from utils.logger import get_logger
from utils.ppo_backtester import create_ppo_backtester

logger = get_logger(__name__)

class BacktesterAgent(BaseAgent):
    """Backtester Agent responsible for backtesting trading strategies using PPO reinforcement learning."""
    
    def __init__(self):
        super().__init__(
            name="Backtester",
            description="Responsible for backtesting trading strategies using PPO reinforcement learning and generating performance reports"
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_by_symbol = input_data.get("data_by_symbol")
            if not data_by_symbol:
                data = input_data.get("data")
                if not data:
                    return self._format_output(
                        status="error",
                        result=None,
                        message="Missing 'data_by_symbol' or 'data'"
                    )
                data_by_symbol = self._split_data_by_symbol(data)

            sentiment_results = input_data.get("sentiment_results", {})
            technical_results = input_data.get("technical_results", {})
            initial_capital = input_data.get("initial_capital", 100000)
            start_date = input_data.get("start_date")
            end_date = input_data.get("end_date")
            backtest_mode = input_data.get("backtest_mode", "ppo")
            total_timesteps = input_data.get("total_timesteps", 100000)
            transaction_cost = input_data.get("transaction_cost", 0.001)
            time_range = input_data.get("time_range", "1y")

            logger.info(f"Backtesting multi-symbol strategy, mode={backtest_mode}")

            results_by_symbol = {}

            for symbol, rows in data_by_symbol.items():
                df = pd.DataFrame(rows)
                if df.empty:
                    continue

                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").set_index("Date")

                if "Symbol" in df.columns:
                    df = df.drop(columns=["Symbol"])

                if start_date and end_date:
                    df = df.loc[start_date:end_date]

                symbol_sentiment = sentiment_results.get("by_symbol", {}).get(symbol, {})
                sentiment_df = self._prepare_sentiment_dataframe(symbol_sentiment, df.index)

                symbol_technical = technical_results.get("results_by_symbol", {}).get(symbol, {})

                if backtest_mode == "ppo":
                    backtest_results = self._run_ppo_backtest(
                        df=df,
                        initial_capital=initial_capital,
                        time_range=time_range,
                        total_timesteps=total_timesteps,
                        transaction_cost=transaction_cost,
                        sentiment_data=sentiment_df,
                        technical_results=symbol_technical
                    )
                else:
                    backtest_results = self._run_traditional_backtest(
                        df,
                        symbol_technical.get("signals", []),
                        initial_capital
                    )

                results_by_symbol[symbol] = {
                    "symbol": symbol,
                    "backtest_mode": backtest_mode,
                    "backtest_results": backtest_results,
                    "performance_metrics": backtest_results.get("portfolio_stats", {}),
                    "summary": backtest_results.get("summary", "")
                }

            result = {
                "initial_capital": initial_capital,
                "backtest_mode": backtest_mode,
                "results_by_symbol": results_by_symbol
            }

            return self._format_output(
                status="success",
                result=result,
                message=f"Successfully backtested strategy using {backtest_mode} mode"
            )

        except Exception as e:
            logger.error(f"Error in BacktesterAgent: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._format_output(
                status="error",
                result=None,
                message=str(e)
            )
    def _split_data_by_symbol(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for row in data:
            symbol = row.get("Symbol", "UNKNOWN")
            results.setdefault(symbol, []).append(row)
        return results

    def _prepare_sentiment_dataframe(self, sentiment_result: Dict[str, Any], price_index) -> pd.DataFrame:
        sentiment_dimensions = sentiment_result.get("sentiment_dimensions", {})
        if not sentiment_dimensions:
            return None

        sent_df = pd.DataFrame(list(sentiment_dimensions.values()))
        if sent_df.empty:
            return None

        sent_df["trade_date"] = pd.to_datetime(sent_df["trade_date"]).dt.normalize()
        sent_df = sent_df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

        numeric_cols = sent_df.select_dtypes(include=["number"]).columns.tolist()
        sent_df = sent_df[["trade_date"] + numeric_cols].set_index("trade_date")

        aligned = sent_df.reindex(pd.to_datetime(price_index).normalize())
        aligned.index = price_index
        aligned = aligned.ffill(limit=1)

        valid_mask = ~aligned.isna().all(axis=1)
        aligned = aligned.loc[valid_mask]

        return aligned
    
    def _run_ppo_backtest(self, df: pd.DataFrame, time_range: str, initial_capital: float, total_timesteps: int, transaction_cost: float,
                    sentiment_data: pd.DataFrame = None, technical_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run backtest using PPO reinforcement learning with data from multiple sources.
        
        Args:
            df: DataFrame containing price data from Data Engineer.
            initial_capital: Initial capital for backtesting.
            total_timesteps: Total timesteps for PPO training.
            transaction_cost: Transaction cost percentage.
            sentiment_data: Sentiment analysis results from Sentiment Analyst.
            technical_results: Technical analysis results from Technical Analyst.
        
        Returns:
            Dictionary containing PPO backtest results.
        """
        logger.info(f"Running PPO backtest with {total_timesteps} timesteps, using multi-source data")
        
        # Create PPO backtester instance with multi-source data
        ppo_backtester = create_ppo_backtester(
            data=df,
            time_range=time_range,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            sentiment_data=sentiment_data,
            technical_results=technical_results
        )
        
        # Train the PPO model
        training_results = ppo_backtester.train(
            total_timesteps=total_timesteps,
            verbose=1
        )
        
        if training_results["status"] != "success":
            logger.error(f"PPO training failed: {training_results['message']}")
            return {
                "status": "error",
                "message": training_results["message"]
            }
        
        # Run backtest with the trained model
        backtest_results = ppo_backtester.backtest()
        
        if backtest_results["status"] != "success":
            logger.error(f"PPO backtest failed: {backtest_results['message']}")
            return {
                "status": "error",
                "message": backtest_results["message"]
            }
        
        # Generate trading signals
        signals = ppo_backtester.generate_signals()
        backtest_results["generated_signals"] = signals
        
        logger.info(f"PPO backtest completed. Final portfolio value: ${backtest_results['portfolio_stats']['final_portfolio_value']:.2f}")
        
        return backtest_results
    
    def _run_traditional_backtest(self, df: pd.DataFrame, signals: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
        """
        Run traditional backtest simulation.
        
        Args:
            df: DataFrame containing price data.
            signals: List of trading signals.
            initial_capital: Initial capital for backtesting.
        
        Returns:
            Dictionary containing traditional backtest results.
        """
        # Create a copy of the data
        backtest_df = df.copy()
        backtest_df["Signal"] = 0  # 0 = hold, 1 = buy, -1 = sell
        backtest_df["Position"] = 0  # Number of shares held
        backtest_df["Portfolio Value"] = initial_capital
        backtest_df["Cash"] = initial_capital
        backtest_df["Returns"] = 0.0
        
        # Convert signals to DataFrame for easier processing
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df["date"] = pd.to_datetime(signals_df["date"])
            
            # Map signals to the backtest dataframe
            for _, signal in signals_df.iterrows():
                date = signal["date"]
                if date in backtest_df.index:
                    backtest_df.loc[date, "Signal"] = 1 if signal["signal"] == "buy" else -1
        
        # Simulate trading
        shares = 0
        cash = initial_capital
        
        for i, row in backtest_df.iterrows():
            # Calculate returns
            if i > backtest_df.index[0]:
                backtest_df.loc[i, "Returns"] = (row["Close"] - backtest_df.loc[backtest_df.index[i-1], "Close"]) / backtest_df.loc[backtest_df.index[i-1], "Close"]
            
            # Execute signals
            if backtest_df.loc[i, "Signal"] == 1:  # Buy signal
                shares_to_buy = cash // row["Close"]
                cost = shares_to_buy * row["Close"]
                shares += shares_to_buy
                cash -= cost
            elif backtest_df.loc[i, "Signal"] == -1:  # Sell signal
                cash += shares * row["Close"]
                shares = 0
            
            # Update portfolio value
            portfolio_value = cash + (shares * row["Close"])
            backtest_df.loc[i, "Position"] = shares
            backtest_df.loc[i, "Cash"] = cash
            backtest_df.loc[i, "Portfolio Value"] = portfolio_value
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(backtest_df)
        
        # Generate summary
        summary = self._generate_summary(performance_metrics)
        
        return {
            "status": "success",
            "message": "Traditional backtest completed",
            "backtest_df": backtest_df,
            "portfolio_stats": performance_metrics,
            "summary": summary
        }
    
    def _calculate_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_results: DataFrame containing backtest results.
        
        Returns:
            Dictionary containing performance metrics.
        """
        portfolio_values = backtest_results["Portfolio Value"]
        returns = backtest_results["Returns"]
        
        # Calculate basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0] * 100
        annualized_return = (1 + total_return / 100) ** (252 / len(portfolio_values)) - 1
        
        # Calculate volatility
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate win rate (simple implementation)
        buy_signals = backtest_results[backtest_results["Signal"] == 1]
        win_trades = 0
        total_trades = 0
        
        for i, buy_date in enumerate(buy_signals.index):
            # Find next sell signal or end of data
            sell_signals_after_buy = backtest_results.loc[buy_date:][backtest_results["Signal"] == -1]
            if not sell_signals_after_buy.empty:
                sell_date = sell_signals_after_buy.index[0]
                buy_price = backtest_results.loc[buy_date, "Close"]
                sell_price = backtest_results.loc[sell_date, "Close"]
                
                if sell_price > buy_price:
                    win_trades += 1
                total_trades += 1
        
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return * 100, 2),
            "annualized_volatility": round(annualized_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": total_trades - win_trades
        }
    
    def _generate_summary(self, performance_metrics: Dict[str, Any]) -> str:
        """
        Generate a summary of the backtest results.
        
        Args:
            performance_metrics: Dictionary containing performance metrics.
        
        Returns:
            Summary of backtest results.
        """
        summary = f"""Traditional Backtest Summary:
- Total Return: {performance_metrics['total_return']}%
- Annualized Return: {performance_metrics['annualized_return']}%
- Annualized Volatility: {performance_metrics['annualized_volatility']}%
- Sharpe Ratio: {performance_metrics['sharpe_ratio']}
- Maximum Drawdown: {performance_metrics['max_drawdown']}%
- Win Rate: {performance_metrics['win_rate']}% ({performance_metrics['win_trades']}/{performance_metrics['total_trades']} trades)
        """
        
        return summary.strip()
