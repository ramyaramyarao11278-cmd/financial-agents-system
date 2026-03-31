from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Base response model
class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: str = Field(..., description="Status of the request (success or error)")
    message: str = Field(..., description="Message describing the result")
    data: Optional[Any] = Field(None, description="Response data")

# Data models
class StockDataPoint(BaseModel):
    """Model representing a single stock data point."""
    Date: str = Field(..., description="Date of the data point")
    Open: float = Field(..., description="Opening price")
    High: float = Field(..., description="Highest price")
    Low: float = Field(..., description="Lowest price")
    Close: float = Field(..., description="Closing price")
    Volume: int = Field(..., description="Trading volume")

class DataFetchRequest(BaseModel):
    """Model for data fetching requests."""
    symbol: Optional[str] = Field(None, description="Stock symbol")
    period: str = Field(default="1y", description="Time period (e.g., '1d', '1mo', '1y')")
    interval: str = Field(default="1d", description="Data interval (e.g., '1m', '15m', '1d')")

class DataFetchResponse(BaseModel):
    """Model for data fetching responses."""
    symbol: str = Field(..., description="Stock symbol")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the data")
    data: List[StockDataPoint] = Field(..., description="Stock data points")

# Sentiment analysis models
class NewsArticle(BaseModel):
    """Model representing a news article."""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    source: str = Field(..., description="News source")
    date: str = Field(..., description="Publication date")

class SentimentRequest(BaseModel):
    """Model for sentiment analysis requests."""
    symbol: str = Field(..., description="Stock symbol")
    news: List[NewsArticle] = Field(..., description="List of news articles")
    timeframe: str = Field(default="recent", description="Timeframe for analysis")

class SentimentResult(BaseModel):
    """Model for sentiment analysis results."""
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(..., description="Timeframe for analysis")
    sentiment_score: float = Field(..., description="Overall sentiment score (-1 to 1)")
    sentiment_classification: str = Field(..., description="Sentiment classification (negative, neutral, positive)")
    sentiment_breakdown: List[Dict[str, Any]] = Field(..., description="Breakdown of sentiment by article")
    summary: str = Field(..., description="Summary of sentiment analysis")

# Technical analysis models
class TechnicalAnalysisRequest(BaseModel):
    """Model for technical analysis requests."""
    symbol: str = Field(..., description="Stock symbol")
    data: List[StockDataPoint] = Field(..., description="Historical price data")
    indicators: Optional[List[str]] = Field(default=None, description="List of indicators to calculate")

class TradingSignal(BaseModel):
    """Model representing a trading signal."""
    date: str = Field(..., description="Date of the signal")
    signal: str = Field(..., description="Signal type (buy, sell, hold)")
    indicator: str = Field(..., description="Indicator generating the signal")
    value: float = Field(..., description="Indicator value")
    reason: str = Field(..., description="Reason for the signal")

class TechnicalAnalysisResult(BaseModel):
    """Model for technical analysis results."""
    symbol: str = Field(..., description="Stock symbol")
    indicators: Dict[str, Any] = Field(..., description="Calculated technical indicators")
    signals: List[TradingSignal] = Field(..., description="Generated trading signals")
    trend_analysis: Dict[str, Any] = Field(..., description="Overall trend analysis")

# Backtesting models
class BacktestRequest(BaseModel):
    """Model for backtesting requests."""
    symbol: str = Field(..., description="Stock symbol")
    data: List[StockDataPoint] = Field(..., description="Historical price data")
    signals: List[TradingSignal] = Field(..., description="Trading signals to backtest")
    initial_capital: float = Field(default=100000, description="Initial capital for backtesting")
    start_date: Optional[str] = Field(default=None, description="Start date for backtesting")
    end_date: Optional[str] = Field(default=None, description="End date for backtesting")

class PerformanceMetrics(BaseModel):
    """Model for backtest performance metrics."""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    annualized_volatility: float = Field(..., description="Annualized volatility percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    total_trades: int = Field(..., description="Total number of trades")
    win_trades: int = Field(..., description="Number of winning trades")
    loss_trades: int = Field(..., description="Number of losing trades")

class BacktestResult(BaseModel):
    """Model for backtest results."""
    symbol: str = Field(..., description="Stock symbol")
    initial_capital: float = Field(..., description="Initial capital used")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    summary: str = Field(..., description="Summary of backtest results")

# Workflow models
class WorkflowRequest(BaseModel):
    """Model for workflow requests."""
    symbol: Optional[str] = Field(None, description="Stock symbol")
    period: str = Field(default="1y", description="Time period for analysis")
    interval: str = Field(default="1d", description="Data interval")
    initial_capital: float = Field(default=100000, description="Initial capital for backtesting")

class WorkflowResult(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Single symbol if applicable")
    status: str = Field(..., description="Workflow status")
    data_result: Optional[Dict[str, Any]] = Field(default=None, description="Data fetching results")
    sentiment_result: Optional[Dict[str, Any]] = Field(default=None, description="Sentiment analysis results")
    technical_result: Optional[Dict[str, Any]] = Field(default=None, description="Technical analysis results")
    backtest_result: Optional[Dict[str, Any]] = Field(default=None, description="Backtest results")
    error: Optional[str] = Field(default=None, description="Error message if any")