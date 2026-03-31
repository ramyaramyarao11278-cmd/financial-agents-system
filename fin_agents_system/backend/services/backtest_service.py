from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from agents.backtester import BacktesterAgent
from config import config
from models.database import BacktestResult as BacktestResultModel
from utils.logger import get_logger

logger = get_logger(__name__)

class BacktestService:
    """Service for managing backtesting operations."""
    
    def __init__(self):
        self.backtester = BacktesterAgent()
        logger.info("Initialized BacktestService")
    
    def run_backtest(self, data: List[Dict[str, Any]], signals: List[Dict[str, Any]], symbol: str = None,
                    initial_capital: float = 100000, start_date: str = None, end_date: str = None, 
                    db: Session = None) -> Dict[str, Any]:
        """
        Run a backtest and optionally store the results.
        
        Args:
            symbol: Stock symbol (optional)
            data: Historical price data
            signals: Trading signals
            initial_capital: Initial capital for backtesting
            start_date: Start date for backtesting
            end_date: End date for backtesting
            db: Database session (optional)
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # If symbol is not provided, use an empty string
            symbol = symbol or ""
            logger.info(f"Running backtest for {symbol}")
            
            # Run backtest using the agent
            result = self.backtester.run({
                "data": data,
                "signals": signals,
                "initial_capital": initial_capital,
                "start_date": start_date,
                "end_date": end_date
            })
            
            if result["status"] != "success":
                logger.error(f"Backtest failed: {result['message']}")
                return result
            
            # Store results in database if session is provided
            if db:
                self._store_backtest_result(result, symbol, db)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _store_backtest_result(self, result: Dict[str, Any], symbol: str, db: Session) -> None:
        """
        Store backtest result in the database.
        
        Args:
            result: Backtest result dictionary
            symbol: Stock symbol
            db: Database session
        """
        try:
            logger.info(f"Storing backtest result for {symbol} in database")
            
            # Extract relevant data from result
            backtest_data = result["result"]
            performance_metrics = backtest_data["performance_metrics"]
            
            # Parse dates from data
            data_dates = [datetime.strptime(item["Date"], "%Y-%m-%d %H:%M:%S") if " " in item["Date"] 
                         else datetime.strptime(item["Date"], "%Y-%m-%d") for item in backtest_data["backtest_results"]]
            
            start_date = min(data_dates)
            end_date = max(data_dates)
            
            # Create database model
            backtest_model = BacktestResultModel(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=backtest_data["initial_capital"],
                total_return=performance_metrics["total_return"],
                annualized_return=performance_metrics["annualized_return"],
                annualized_volatility=performance_metrics["annualized_volatility"],
                sharpe_ratio=performance_metrics["sharpe_ratio"],
                max_drawdown=performance_metrics["max_drawdown"],
                win_rate=performance_metrics["win_rate"],
                total_trades=performance_metrics["total_trades"],
                win_trades=performance_metrics["win_trades"],
                loss_trades=performance_metrics["loss_trades"],
                summary=backtest_data["summary"],
                backtest_data=result
            )
            
            # Save to database
            db.add(backtest_model)
            db.commit()
            db.refresh(backtest_model)
            
            logger.info(f"Successfully stored backtest result with ID: {backtest_model.id}")
            
        except Exception as e:
            logger.error(f"Error storing backtest result: {str(e)}")
            db.rollback()
            raise
    
    def get_backtest_results(self, symbol: str = None, start_date: str = None, end_date: str = None, 
                           db: Session = None) -> List[Dict[str, Any]]:
        """
        Get backtest results from the database.
        
        Args:
            symbol: Stock symbol (optional)
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            db: Database session
            
        Returns:
            List of backtest results
        """
        try:
            logger.info("Getting backtest results from database")
            
            if not db:
                logger.warning("No database session provided, returning empty list")
                return []
            
            # Build query
            query = db.query(BacktestResultModel)
            
            # Apply filters
            if symbol:
                query = query.filter(BacktestResultModel.symbol == symbol)
            
            if start_date:
                query = query.filter(BacktestResultModel.start_date >= datetime.strptime(start_date, "%Y-%m-%d"))
            
            if end_date:
                query = query.filter(BacktestResultModel.end_date <= datetime.strptime(end_date, "%Y-%m-%d"))
            
            # Execute query
            results = query.order_by(BacktestResultModel.end_date.desc()).all()
            
            # Convert to list of dictionaries
            backtest_results = []
            for result in results:
                backtest_results.append({
                    "id": result.id,
                    "symbol": result.symbol,
                    "start_date": result.start_date.strftime("%Y-%m-%d"),
                    "end_date": result.end_date.strftime("%Y-%m-%d"),
                    "initial_capital": result.initial_capital,
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "annualized_volatility": result.annualized_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "win_trades": result.win_trades,
                    "loss_trades": result.loss_trades,
                    "summary": result.summary,
                    "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "backtest_data": result.backtest_data
                })
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {str(e)}")
            raise
    
    def get_backtest_result_by_id(self, result_id: int, db: Session = None) -> Dict[str, Any]:
        """
        Get a specific backtest result by ID.
        
        Args:
            result_id: Backtest result ID
            db: Database session
            
        Returns:
            Dictionary containing backtest result
        """
        try:
            logger.info(f"Getting backtest result with ID: {result_id}")
            
            if not db:
                logger.warning("No database session provided, cannot get result")
                return {"status": "error", "message": "No database session provided"}
            
            # Get result from database
            result = db.query(BacktestResultModel).filter(BacktestResultModel.id == result_id).first()
            
            if not result:
                logger.warning(f"Backtest result with ID {result_id} not found")
                return {"status": "error", "message": f"Backtest result with ID {result_id} not found"}
            
            # Convert to dictionary
            return {
                "status": "success",
                "data": {
                    "id": result.id,
                    "symbol": result.symbol,
                    "start_date": result.start_date.strftime("%Y-%m-%d"),
                    "end_date": result.end_date.strftime("%Y-%m-%d"),
                    "initial_capital": result.initial_capital,
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "annualized_volatility": result.annualized_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "win_trades": result.win_trades,
                    "loss_trades": result.loss_trades,
                    "summary": result.summary,
                    "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "backtest_data": result.backtest_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting backtest result by ID: {str(e)}")
            raise
    
    def delete_backtest_result(self, result_id: int, db: Session = None) -> Dict[str, Any]:
        """
        Delete a backtest result by ID.
        
        Args:
            result_id: Backtest result ID
            db: Database session
            
        Returns:
            Dictionary containing deletion result
        """
        try:
            logger.info(f"Deleting backtest result with ID: {result_id}")
            
            if not db:
                logger.warning("No database session provided, cannot delete result")
                return {"status": "error", "message": "No database session provided"}
            
            # Delete result
            deleted_count = db.query(BacktestResultModel).filter(BacktestResultModel.id == result_id).delete()
            db.commit()
            
            if deleted_count == 0:
                logger.warning(f"Backtest result with ID {result_id} not found")
                return {"status": "error", "message": f"Backtest result with ID {result_id} not found"}
            
            return {
                "status": "success",
                "message": f"Deleted backtest result with ID {result_id}",
                "deleted_count": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting backtest result: {str(e)}")
            db.rollback()
            raise

# Singleton instance
_backtest_service = None

def get_backtest_service() -> BacktestService:
    """Get singleton instance of BacktestService."""
    global _backtest_service
    if _backtest_service is None:
        _backtest_service = BacktestService()
    return _backtest_service
