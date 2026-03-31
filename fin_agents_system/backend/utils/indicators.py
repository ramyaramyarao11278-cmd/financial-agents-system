from typing import Tuple

import numpy as np
import pandas as pd


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of prices
        window: Number of periods to calculate SMA for
        
    Returns:
        Series containing SMA values
    """
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Series of prices
        window: Number of periods to calculate EMA for
        
    Returns:
        Series containing EMA values
    """
    return prices.ewm(span=window, adjust=False).mean()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of prices
        window: Number of periods to calculate RSI for (default: 14)
        
    Returns:
        Series containing RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Avoid division by zero
    loss = loss.replace(0, 1e-10)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Series of prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        
    Returns:
        Tuple containing:
        - macd_line: MACD line values
        - signal_line: Signal line values
        - histogram: MACD histogram values
    """
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of prices
        window: Number of periods to calculate Bollinger Bands for (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        
    Returns:
        Tuple containing:
        - upper_band: Upper Bollinger Band values
        - middle_band: Middle Bollinger Band (SMA) values
        - lower_band: Lower Bollinger Band values
    """
    middle_band = calculate_sma(prices, window)
    rolling_std = prices.rolling(window=window).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_stochastics(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame containing 'High', 'Low', and 'Close' columns
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        
    Returns:
        Tuple containing:
        - %K line values
        - %D line values
    """
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def calculate_momentum(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Momentum.
    
    Args:
        prices: Series of prices
        window: Number of periods to calculate momentum for (default: 14)
        
    Returns:
        Series containing momentum values
    """
    return prices.diff(window)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame containing 'High', 'Low', and 'Close' columns
        window: Number of periods to calculate ATR for (default: 14)
        
    Returns:
        Series containing ATR values
    """
    # Calculate true range components
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        df: DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns
        window: Number of periods to calculate MFI for (default: 14)
        
    Returns:
        Series containing MFI values
    """
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate money flow
    money_flow = typical_price * df['Volume']
    
    # Determine positive and negative money flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    
    # Calculate 14-day sums
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    # Avoid division by zero
    negative_mf = negative_mf.replace(0, 1e-10)
    
    # Calculate money flow ratio
    mfi_ratio = positive_mf / negative_mf
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + mfi_ratio))
    
    return mfi


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: DataFrame containing 'Close' and 'Volume' columns
        
    Returns:
        Series containing OBV values
    """
    obv = []
    obv_value = 0
    
    for i in range(len(df)):
        if i == 0:
            obv_value = df['Volume'].iloc[i]
        else:
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv_value += df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv_value -= df['Volume'].iloc[i]
            # If prices are equal, OBV remains the same
        obv.append(obv_value)
    
    return pd.Series(obv, index=df.index, name='OBV')


def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        df: DataFrame containing 'High', 'Low', and 'Close' columns
        window: Number of periods to calculate CCI for (default: 20)
        
    Returns:
        Series containing CCI values
    """
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate SMA of typical price
    tp_sma = typical_price.rolling(window=window).mean()
    
    # Calculate mean deviation
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    # Avoid division by zero
    mean_deviation = mean_deviation.replace(0, 1e-10)
    
    # Calculate CCI
    cci = (typical_price - tp_sma) / (0.015 * mean_deviation)
    
    return cci

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe.
    """

    # Moving averages
    df["SMA_5"] = calculate_sma(df["Close"], 5)
    df["SMA_20"] = calculate_sma(df["Close"], 20)

    df["EMA_12"] = calculate_ema(df["Close"], 12)
    df["EMA_26"] = calculate_ema(df["Close"], 26)

    # RSI
    df["RSI"] = calculate_rsi(df["Close"], 14)

    # MACD
    macd, signal, hist = calculate_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = hist

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df["Close"])
    df["BB_UPPER"] = upper
    df["BB_MIDDLE"] = middle
    df["BB_LOWER"] = lower

    # Stochastic
    k, d = calculate_stochastics(df)
    df["STOCH_K"] = k
    df["STOCH_D"] = d

    # Momentum
    df["MOMENTUM"] = calculate_momentum(df["Close"])

    # ATR
    df["ATR"] = calculate_atr(df)

    # MFI
    if "Volume" in df.columns:
        df["MFI"] = calculate_mfi(df)

    # OBV
    if "Volume" in df.columns:
        df["OBV"] = calculate_obv(df)

    # CCI
    df["CCI"] = calculate_cci(df)

    # 删除 NaN
    df = df.fillna(method="bfill").fillna(method="ffill")

    return df
