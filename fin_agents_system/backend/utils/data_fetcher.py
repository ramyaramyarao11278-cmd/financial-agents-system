from typing import Any, Dict, Optional
import os

import pandas as pd
import yfinance as yf

from utils.logger import get_logger

logger = get_logger(__name__)

# 基于脚本位置计算 documents 目录路径（兼容任何部署位置）
_DOCUMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "documents"))

def fetch_stock_data(
    symbol: str = None,
    time_range: str = "1y",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical stock data for CSI100 or CSI300 indices from local Excel files.
    """

    # Only allow CSI100 / CSI300
    if not symbol or symbol not in ["CSI100", "CSI300"]:
        logger.warning(
            f"Symbol {symbol} is not supported or not provided. Using CSI300 as default."
        )
        symbol = "CSI300"

    logger.info(f"Fetching local data for {symbol}")

    # Excel file mapping
    file_map = {
        "CSI300": os.path.join(_DOCUMENTS_DIR, "沪深300历史数据.xlsx"),
        "CSI100": os.path.join(_DOCUMENTS_DIR, "中证100指数历史数据.xlsx"),
    }

    file_path = file_map[symbol]

    try:
        # 读取 Excel
        df = pd.read_excel(file_path)

        # 转换日期
        df["日期"] = pd.to_datetime(df["日期"])

        # 重命名列以匹配原系统格式
        df = df.rename(
            columns={
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
                "日期": "Date",
            }
        )

        # 保留需要的列
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # 设置 index
        df = df.set_index("Date")

        # 按时间排序
        df = df.sort_index()
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # 根据 time_range 过滤
        if start_date is None and end_date is None:

            range_map = {
                "7d": "7D",
                "1m": "30D",
                "3m": "90D",
                "1y": "365D",
                "2y": "730D"
            }

            if time_range in range_map:
                df = df.last(range_map[time_range])

        logger.info(f"Loaded {len(df)} rows from local file for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Error loading local data: {str(e)}")
        logger.info("Generating simulated data instead")

        return generate_simulated_data(symbol, time_range, interval)


def generate_simulated_data(symbol: str,  time_range: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Generate simulated stock/index data for testing purposes.
    
    Args:
        symbol: Stock or index symbol
        period: Time period (e.g., "1y")
        interval: Data interval (e.g., "1d")
        
    Returns:
        DataFrame containing simulated historical data
    """
    from datetime import datetime, timedelta

    import numpy as np

    range_map = {
        "7d": 7,
        "1m": 30,
        "3m": 90,
        "1y": 252,
        "2y": 500
    }

    target_periods = range_map.get(time_range, 60)
    
    # Generate dates first to get exact number of periods
    end_date = datetime.now()
    start_date = end_date - timedelta(days=target_periods + 20)  # Add buffer
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    actual_periods = min(len(dates), target_periods)
    dates = dates[-actual_periods:]  # Use most recent dates
    
    # Generate base price with upward trend
    base_price = 4000 if symbol in ["CSI100", "CSI300"] else 100
    trend = np.linspace(0, 1000, actual_periods) if symbol in ["CSI100", "CSI300"] else np.linspace(0, 20, actual_periods)
    volatility = 50 if symbol in ["CSI100", "CSI300"] else 2
    
    # Generate price series
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility, actual_periods)
    close_prices = base_price + trend + np.cumsum(returns)
    
    # Generate other price columns with exact period match
    # Start with open prices (first open = base price, then derived from previous close)
    open_prices = np.zeros(actual_periods)
    open_prices[0] = base_price
    if actual_periods > 1:
        open_prices[1:] = close_prices[:-1] + np.random.normal(0, volatility * 0.5, actual_periods - 1)
    
    # Generate high and low prices
    high_prices = np.maximum(open_prices, close_prices) + np.random.normal(0, volatility * 0.3, actual_periods)
    low_prices = np.minimum(open_prices, close_prices) - np.random.normal(0, volatility * 0.3, actual_periods)
    
    # Generate volume
    volumes = np.random.randint(1000000, 10000000, actual_periods) if symbol in ["CSI100", "CSI300"] else np.random.randint(100000, 1000000, actual_periods)
    
    # Create DataFrame with all columns having the same length
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    logger.info(f"Generated {len(df)} simulated data points for {symbol}")
    return df

def fetch_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Fetch general information about a stock.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing stock information
    """
    try:
        logger.info(f"Fetching stock info for {symbol}")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant information
        relevant_info = {
            "symbol": symbol,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "previous_close": info.get("previousClose", 0),
            "open": info.get("open", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "volume": info.get("volume", 0),
            "average_volume": info.get("averageVolume", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "eps": info.get("trailingEps", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0)
        }
        
        return relevant_info
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        return {"symbol": symbol, "error": str(e)}

def fetch_multiple_stocks(symbols: list, time_range: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stocks.
    
    Args:
        symbols: List of stock symbols
        period: Time period to fetch data for
        interval: Data interval
        
    Returns:
        Dictionary where keys are symbols and values are DataFrames containing stock data
    """
    results = {}
    
    for symbol in symbols:
        df = fetch_stock_data(symbol, time_range, interval)
        results[symbol] = df
    
    return results

def fetch_dividends(symbol: str, time_range: str = "1y") -> pd.DataFrame:
    """
    Fetch dividend history for a stock.
    
    Args:
        symbol: Stock symbol
        period: Time period to fetch dividends for
        
    Returns:
        DataFrame containing dividend history
    """
    try:
        logger.info(f"Fetching dividend data for {symbol}")
        
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
        
        # Filter by period if needed
        if time_range == "1y":
            dividends = dividends.last("1y")
        elif time_range == "5y":
            dividends = dividends.last("5y")
        elif time_range == "max":
            pass  # Return all dividends
        
        return dividends
        
    except Exception as e:
        logger.error(f"Error fetching dividends for {symbol}: {str(e)}")
        return pd.Series(dtype='float64')

def fetch_split_history(symbol: str) -> pd.DataFrame:
    """
    Fetch stock split history.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        DataFrame containing split history
    """
    try:
        logger.info(f"Fetching split history for {symbol}")
        
        ticker = yf.Ticker(symbol)
        splits = ticker.splits
        
        return splits
        
    except Exception as e:
        logger.error(f"Error fetching split history for {symbol}: {str(e)}")
        return pd.Series(dtype='float64')
