from typing import Any, Dict, List
import asyncio
import os
import pandas as pd
from agents.backtester import BacktesterAgent
from agents.data_engineer import DataEngineerAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session
from utils.logger import get_logger

# Global callback function to send agent execution updates
default_agent_update_callback = None

def set_agent_update_callback(callback):
    """Set the callback function to send agent updates."""
    global default_agent_update_callback
    default_agent_update_callback = callback

def send_agent_update(agent_name: str, status: str, message: str = ""):
    """Send an update about agent execution."""
    if default_agent_update_callback:
        try:
            # 直接调用回调函数，让它处理事件循环
            default_agent_update_callback(agent_name, status, message)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calling agent update callback: {e}")

logger = get_logger(__name__)

# Initialize agents
data_engineer = DataEngineerAgent()
sentiment_analyst = SentimentAnalystAgent()
technical_analyst = TechnicalAnalystAgent()
backtester = BacktesterAgent()

# Define workflow state as a dataclass for better type safety
from dataclasses import dataclass

@dataclass
class WorkflowState:
    symbol: str = ""
    url: str = ""
    time_range: str = "7d"
    interval: str = "1d"
    sentiment_files: Dict[str, str] = None

    data: List[Dict[str, Any]] = None
    data_by_symbol: Dict[str, List[Dict[str, Any]]] = None
    news: List[Dict[str, Any]] = None
    sentiment_data_by_symbol: Dict[str, List[Dict[str, Any]]] = None

    sentiment_results: Dict[str, Any] = None
    technical_results: Dict[str, Any] = None
    backtest_results: Dict[str, Any] = None

    db_session: Session = None
    error: str = ""
    status: str = "initialized"

    def __post_init__(self):
        if self.data is None:
            self.data = []
        if self.data_by_symbol is None:
            self.data_by_symbol = {}
        if self.news is None:
            self.news = []
        if self.sentiment_data_by_symbol is None:
            self.sentiment_data_by_symbol = {}
        if self.sentiment_results is None:
            self.sentiment_results = {}
        if self.technical_results is None:
            self.technical_results = {}
        if self.backtest_results is None:
            self.backtest_results = {}
        if self.sentiment_files is None:
            self.sentiment_files = {}
    

# Define workflow nodes
async def data_engineer_node(state: WorkflowState) -> Dict[str, Any]:
    agent_name = "数据工程师"
    send_agent_update(agent_name, "running", "正在加载本地价格与情感数据...")
    logger.info("Data Engineer: Loading local price and sentiment files")

    await asyncio.sleep(1)

    _documents_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "documents"))
    result = data_engineer.run({
        "url": state.url,
        "time_range": state.time_range,
        "interval": state.interval,
        "db": state.db_session,
        "sentiment_files": {
            "CSI100": os.path.join(_documents_dir, "CSI100.csv"),
            "CSI300": os.path.join(_documents_dir, "CSI300.csv"),
        }
    })

    await asyncio.sleep(1)

    if result["status"] == "success":
        send_agent_update(agent_name, "completed", "本地数据加载完成")
        return {
            "data": result["result"]["data"],
            "data_by_symbol": result["result"]["data_by_symbol"],
            "sentiment_data_by_symbol": result["result"]["sentiment_data_by_symbol"],
            "news": [],
            "status": "data_fetched"
        }

    send_agent_update(agent_name, "failed", f"数据获取失败: {result['message']}")
    return {
        "error": result["message"],
        "status": "error"
    }

async def sentiment_analyst_node(state: WorkflowState) -> Dict[str, Any]:
    agent_name = "情感分析师"
    send_agent_update(agent_name, "running", "正在读取预计算情感特征...")
    logger.info("Sentiment Analyst: Loading precomputed sentiment features")

    await asyncio.sleep(1)

    if state.sentiment_data_by_symbol:
        result = sentiment_analyst.run({
            "sentiment_data_by_symbol": state.sentiment_data_by_symbol,
            "timeframe": "historical"
        })
    else:
        result = sentiment_analyst.run({
            "news": state.news,
            "timeframe": "recent"
        })

    await asyncio.sleep(1)

    if result["status"] == "success":
        if state.db_session and result["result"].get("by_symbol"):
            from datetime import date, datetime
            from models.database import SentimentAnalysis

            def make_serializable(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif hasattr(obj, "__dict__"):
                    return obj.__dict__
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                return obj

            for symbol, symbol_result in result["result"]["by_symbol"].items():
                sentiment_record = SentimentAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    sentiment_score=symbol_result.get("sentiment_score", 0.0),
                    sentiment_classification=symbol_result.get("sentiment_classification", "neutral"),
                    sentiment_breakdown=make_serializable(symbol_result.get("sentiment_breakdown", [])),
                    summary=symbol_result.get("summary", ""),
                    source_data=make_serializable(symbol_result.get("sentiment_dimensions", {}))
                )
                state.db_session.add(sentiment_record)

            state.db_session.commit()

        send_agent_update(agent_name, "completed", "情感特征读取完成")
        return {
            "sentiment_results": result["result"],
            "status": "sentiment_analyzed"
        }

    send_agent_update(agent_name, "failed", f"情感分析失败: {result['message']}")
    return {
        "error": result["message"],
        "status": "error"
    }

async def technical_analysis_node(state: WorkflowState) -> Dict[str, Any]:
    agent_name = "技术分析师"
    send_agent_update(agent_name, "running", "正在进行双指数技术分析与预测...")
    logger.info("Technical Analysis Expert: Performing technical analysis")

    await asyncio.sleep(1)

    result = technical_analyst.run({
        "data": state.data,
        "data_by_symbol": state.data_by_symbol,
        "sentiment_results": state.sentiment_results
    })

    await asyncio.sleep(1)

    if result["status"] == "success":
        if state.db_session:
            from datetime import date, datetime
            from models.database import TechnicalAnalysis

            def make_serializable(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif hasattr(obj, "__dict__"):
                    return obj.__dict__
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                return obj

            for symbol, symbol_result in result["result"].get("results_by_symbol", {}).items():
                technical_record = TechnicalAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    indicators=make_serializable(symbol_result.get("indicators", {})),
                    signals=make_serializable(symbol_result.get("signals", [])),
                    trend_analysis=make_serializable(symbol_result.get("trend_analysis", {}))
                )
                state.db_session.add(technical_record)

            state.db_session.commit()

        send_agent_update(agent_name, "completed", "技术分析完成")
        return {
            "technical_results": result["result"],
            "status": "technical_analyzed"
        }

    send_agent_update(agent_name, "failed", f"技术分析失败: {result['message']}")
    return {
        "error": result["message"],
        "status": "error"
    }

async def backtester_node(state: WorkflowState) -> Dict[str, Any]:
    agent_name = "回测专家"
    send_agent_update(agent_name, "running", "正在进行双指数策略回测...")
    logger.info("Backtester: Backtesting strategy")

    await asyncio.sleep(1)

    result = backtester.run({
        "data": state.data,
        "data_by_symbol": state.data_by_symbol,
        "sentiment_results": state.sentiment_results,
        "technical_results": state.technical_results,
        "backtest_mode": "ppo",
        "time_range": state.time_range,
        "total_timesteps": 5000,
        "transaction_cost": 0.001
    })

    await asyncio.sleep(1)

    if result["status"] == "success":
        if state.db_session:
            from datetime import datetime
            from models.database import BacktestResult

            for symbol, symbol_result in result["result"].get("results_by_symbol", {}).items():
                backtest_data = symbol_result.get("backtest_results", {})
                portfolio_stats = symbol_result.get("performance_metrics", {})

                dates = backtest_data.get("backtest_details", {}).get("dates", [])
                if dates:
                    start_date = pd.to_datetime(dates[0])
                    end_date = pd.to_datetime(dates[-1])
                else:
                    start_date = datetime.utcnow()
                    end_date = datetime.utcnow()

                backtest_record = BacktestResult(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=result["result"].get("initial_capital", 100000),
                    total_return=portfolio_stats.get("total_return", 0),
                    annualized_return=portfolio_stats.get("annualized_return", 0),
                    annualized_volatility=portfolio_stats.get("annualized_volatility", 0),
                    sharpe_ratio=portfolio_stats.get("sharpe_ratio", 0),
                    max_drawdown=portfolio_stats.get("max_drawdown", 0),
                    win_rate=portfolio_stats.get("win_rate", 0),
                    total_trades=portfolio_stats.get("total_trades", 0),
                    win_trades=portfolio_stats.get("win_trades", 0),
                    loss_trades=portfolio_stats.get("loss_trades", 0),
                    summary=symbol_result.get("summary", ""),
                    backtest_data=symbol_result
                )
                state.db_session.add(backtest_record)

            state.db_session.commit()

        send_agent_update(agent_name, "completed", "策略回测完成")
        return {
            "backtest_results": result["result"],
            "status": "backtest_completed"
        }

    send_agent_update(agent_name, "failed", f"策略回测失败: {result['message']}")
    return {
        "error": result["message"],
        "status": "error"
    }

def create_workflow():
    """Create and return the financial agents workflow according to the diagram."""
    # Initialize the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes to the workflow
    workflow.add_node("data_engineer", data_engineer_node)
    workflow.add_node("sentiment_analyst", sentiment_analyst_node)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("backtester", backtester_node)
    
    # Set up the workflow edges according to the diagram:
    # 1. Data Engineer -> Sentiment Analyst
    # 2. Sentiment Analyst -> Technical Analysis Expert
    # 3. Technical Analysis Expert -> Backtester
    # 4. Backtester -> END (final node)
    workflow.add_edge("data_engineer", "sentiment_analyst")
    workflow.add_edge("sentiment_analyst", "technical_analysis")
    workflow.add_edge("technical_analysis", "backtester")
    workflow.add_edge("backtester", END)
    
    # Set the entry point
    workflow.set_entry_point("data_engineer")
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow

# Create the workflow instance
financial_workflow = create_workflow()

