import asyncio
import json
import logging
import sys
import os
from typing import Dict, Set

# Set environment variables for proper encoding before any other imports
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Configure logging with a custom formatter that handles Unicode
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Ensure message is encoded as UTF-8 when writing to stream
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(sys.stdout)
    ]
)

import numpy as np
from fastapi import Depends, FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from config import config
from graph.workflow import WorkflowState, financial_workflow
from models.database import Base, init_db
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from utils.logger import get_logger

logger = get_logger(__name__)

# Create database engine
engine = create_engine(config.DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize database
init_db(engine)

app = FastAPI(title="Financial Agents System", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager to handle multiple connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except RuntimeError:
            # 忽略已关闭连接的发送错误
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        except Exception as e:
            # 记录其他发送错误并移除连接
            logger.error(f"WebSocket发送个人消息失败: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        # 创建连接副本以避免迭代时修改集合
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except RuntimeError:
                # 忽略已关闭连接的发送错误
                self.active_connections.remove(connection)
            except Exception as e:
                # 记录其他发送错误并移除连接
                logger.error(f"WebSocket发送消息失败: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await manager.send_personal_message({
            "type": "connection_status",
            "status": "connected",
            "message": "WebSocket连接已建立，等待状态更新..."
        }, websocket)
        
        # Keep connection open without blocking on receive
        # This allows the event loop to handle other tasks and send updates
        while True:
            # Use a small sleep to avoid busy-waiting
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0  # Replace NaN and Inf with 0.0
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Function to clean NaN values from nested dictionaries
def clean_nan_values(data):
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return data
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    # 处理pandas Timestamp对象，转换为字符串以便JSON序列化
    elif hasattr(data, 'isoformat'):
        return data.isoformat()
    return data

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "Financial Agents System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Import the workflow module and set up callback
from graph.workflow import set_agent_update_callback

# Define the agent update callback function
async def agent_update_callback(agent_name: str, status: str, message: str = ""):
    """Callback function to send agent update via WebSocket"""
    update_message = {
        "type": "agent_update",
        "agent_name": agent_name,
        "status": status,
        "message": message
    }
    await manager.broadcast(update_message)

# 后台执行工作流的异步函数
async def run_workflow_background(initial_state):
    """在后台运行工作流并处理结果"""
    try:
        # Run workflow asynchronously
        result = await financial_workflow.ainvoke(initial_state)
        
        # 打印工作流执行结果，用于调试
        # logger.info(f"Workflow execution result: {result}")
        
        # 检查工作流状态，只有当最后一个节点成功执行时才视为成功
        workflow_success = result.get("status") == "backtest_completed" and not result.get("error")
        
        # Clean NaN values from the results to make them JSON-compliant
        cleaned_result = {
            "status": "completed" if workflow_success else "error",
            "message": "Workflow executed successfully" if workflow_success else f"Error: {result.get('error', 'Unknown error')}",
            "results": {
                "sentiment": clean_nan_values(result.get("sentiment_results", {})),
                "technical": clean_nan_values(result.get("technical_results", {})),
                "backtest": clean_nan_values(result.get("backtest_results", {}))
            }
        }
        
        # 打印清理后的结果，用于调试
        # logger.info(f"Cleaned workflow results: {cleaned_result}")
        
        # Save transformer input data to CSV file if available
        import os
        from datetime import datetime

        import pandas as pd
        
        technical_results = result.get("technical_results", {})
        if technical_results:
            # Save to records.csv in documents directory using relative path
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "documents", "records.csv")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # Check if we have transformer input data from technical results
            transformer_input_data = technical_results.get("transformer_input_data", [])
            
            if transformer_input_data:
                # Convert transformer input data to DataFrame
                df = pd.DataFrame(transformer_input_data)
                
                # Append to existing CSV file or create new one
                df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False, encoding='utf-8-sig')
                
                logger.info(f"Transformer input data saved to CSV: {csv_path}")
            else:
                # Fallback: if no transformer input data, check for sentiment dimensions
                sentiment_results = result.get("sentiment_results", {})
                sentiment_dimensions = sentiment_results.get("sentiment_dimensions", {})
                
                if sentiment_dimensions:
                    # Convert sentiment dimensions to DataFrame
                    df = pd.DataFrame(list(sentiment_dimensions.values()))
                    
                    # Append to existing CSV file or create new one
                    df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False, encoding='utf-8-sig')
                    
                    logger.info(f"Sentiment dimensions saved to CSV: {csv_path}")
                else:
                    logger.warning("No transformer input data or sentiment dimensions available to save to CSV")
        
        # 发送工作流完成的WebSocket通知，包含完整分析结果
        await manager.broadcast({
            "type": "analysis_complete",
            "status": "completed",  # 无论工作流内部结果如何，WebSocket通知都标记为completed表示通知已发送
            "message": "工作流执行完成",
            "results": cleaned_result
        })
        
        # 发送智能体更新通知
        if workflow_success:
            await agent_update_callback("系统", "completed", "工作流执行完成")
        else:
            await agent_update_callback("系统", "failed", f"工作流执行失败: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error running workflow in background: {e}")
        import traceback
        traceback.print_exc()
        
        # 构建错误结果
        error_result = {
            "status": "error",
            "message": f"工作流执行失败: {str(e)}",
            "results": {
                "sentiment": {},
                "technical": {},
                "backtest": {}
            }
        }
        
        # 发送工作流失败的WebSocket通知，确保前端收到完成通知
        await manager.broadcast({
            "type": "analysis_complete",
            "status": "completed",  # 无论工作流内部结果如何，WebSocket通知都标记为completed表示通知已发送
            "message": "工作流执行完成",
            "results": error_result
        })
        
        # 发送智能体更新通知
        await agent_update_callback("系统", "failed", f"工作流执行失败: {str(e)}")

@app.post("/run-workflow")
async def run_workflow(
    url: str = Form(...),
    time_range: str = Form("7d"),
    interval: str = Form("1d"),
    db: Session = Depends(get_db)
):
    """
    Run the financial agents workflow in background.
    
    Args:
        url: Initial URL for data crawling
        time_range: Time range for data collection
        interval: Data interval
        db: Database session
    
    Returns:
        Immediate response indicating workflow has started
    """
    # Set the agent update callback with proper event loop handling
    def callback_wrapper(agent_name, status, message):
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Create task on the existing event loop
        loop.create_task(agent_update_callback(agent_name, status, message))
    
    set_agent_update_callback(callback_wrapper)
    
    # Create initial workflow state with db session
    initial_state = WorkflowState()
    initial_state.url = url
    initial_state.time_range = time_range
    initial_state.interval = interval
    initial_state.db_session = db
    
    # 在后台异步执行工作流，不阻塞当前请求
    asyncio.create_task(run_workflow_background(initial_state))
    
    # 立即返回响应，让前端能够实时接收WebSocket消息
    return {
        "status": "running",
        "message": "工作流已启动，正在执行中..."
    }
