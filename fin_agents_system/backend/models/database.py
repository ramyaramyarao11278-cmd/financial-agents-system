from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class StockData(Base):
    """Model for storing stock price data."""
    __tablename__ = "stock_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=True)
    date = Column(DateTime, index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentiment_analyses = relationship("SentimentAnalysis", back_populates="stock_data")
    technical_analyses = relationship("TechnicalAnalysis", back_populates="stock_data")

class SentimentAnalysis(Base):
    """Model for storing sentiment analysis results."""
    __tablename__ = "sentiment_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    sentiment_score = Column(Float, nullable=False)
    sentiment_classification = Column(String(20), nullable=False)
    sentiment_breakdown = Column(JSON, nullable=False)
    summary = Column(Text, nullable=False)
    source_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_data_id = Column(Integer, ForeignKey("stock_data.id"))
    stock_data = relationship("StockData", back_populates="sentiment_analyses")
    backtest_results = relationship("BacktestResult", back_populates="sentiment_analysis")

class TechnicalAnalysis(Base):
    """Model for storing technical analysis results."""
    __tablename__ = "technical_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    indicators = Column(JSON, nullable=False)
    signals = Column(JSON, nullable=False)
    trend_analysis = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_data_id = Column(Integer, ForeignKey("stock_data.id"))
    stock_data = relationship("StockData", back_populates="technical_analyses")
    backtest_results = relationship("BacktestResult", back_populates="technical_analysis")

class BacktestResult(Base):
    """Model for storing backtest results."""
    __tablename__ = "backtest_result"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    annualized_volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    win_trades = Column(Integer, nullable=False)
    loss_trades = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)
    backtest_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentiment_analysis_id = Column(Integer, ForeignKey("sentiment_analysis.id"))
    sentiment_analysis = relationship("SentimentAnalysis", back_populates="backtest_results")
    technical_analysis_id = Column(Integer, ForeignKey("technical_analysis.id"))
    technical_analysis = relationship("TechnicalAnalysis", back_populates="backtest_results")

class WorkflowRun(Base):
    """Model for storing workflow run results."""
    __tablename__ = "workflow_run"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    period = Column(String(20), nullable=False)
    interval = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    error_message = Column(Text, nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow_data = relationship("WorkflowData", back_populates="workflow_run")

class WorkflowData(Base):
    """Model for storing workflow data."""
    __tablename__ = "workflow_data"
    
    id = Column(Integer, primary_key=True, index=True)
    workflow_run_id = Column(Integer, ForeignKey("workflow_run.id"))
    data_type = Column(String(50), nullable=False)  # data, sentiment, technical, backtest
    data_content = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow_run = relationship("WorkflowRun", back_populates="workflow_data")

# Initialize database (to be called from main.py)
def init_db(engine):
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
