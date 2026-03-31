from datetime import datetime
from typing import Any, Dict, List

from config import config
from models.database import StockData as StockDataModel
from sqlalchemy.orm import Session
from utils.data_fetcher import fetch_stock_data
from utils.logger import get_logger

logger = get_logger(__name__)

class DataService:
    """Service for managing financial data."""
    
    def __init__(self):
        logger.info("Initialized DataService")
    
    def fetch_and_store_stock_data(self, symbol: str = None, period: str = "1y", interval: str = "1d", db: Session = None) -> Dict[str, Any]:
        """
        Fetch stock data and store it in the database.
        
        Args:
            symbol: Stock symbol (optional)
            period: Time period (e.g., '1d', '1mo', '1y')
            interval: Data interval (e.g., '1m', '15m', '1d')
            db: Database session (optional)
            
        Returns:
            Dictionary containing fetched data and metadata
        """
        try:
            # If symbol is not provided, use CSI300 as default
            symbol = symbol or "CSI300"
            logger.info(f"Fetching and storing stock data for {symbol}")
            
            # Fetch data from external API
            df = fetch_stock_data(
                symbol=symbol,
                interval=interval,
                start_date="2013-01-01",
                end_date="2025-01-07"
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return {"symbol": symbol, "data": [], "metadata": {"message": "No data available"}}
            
            # Convert to list of dictionaries for API response
            data_list = df.reset_index().to_dict(orient="records")
            
            # Store in database if session is provided
            if db:
                self._store_data_in_db(df, symbol, db)
            
            # Prepare metadata
            metadata = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data_points": len(df),
                "date_range": {
                    "start": str(df.index[0]),
                    "end": str(df.index[-1])
                }
            }
            
            return {
                "symbol": symbol,
                "data": data_list,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching and storing stock data: {str(e)}")
            raise
    
    def _store_data_in_db(self, df: Any, symbol: str, db: Session) -> None:
        """
        Store stock data in the database.
        
        Args:
            df: DataFrame containing stock data
            symbol: Stock symbol
            db: Database session
        """
        try:
            logger.info(f"Storing data for {symbol} in database")
            
            # Convert DataFrame to database models
            stock_data_models = []
            for index, row in df.iterrows():
                stock_data = StockDataModel(
                    symbol=symbol,
                    date=index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index,
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    volume=row['Volume']
                )
                stock_data_models.append(stock_data)
            
            # Bulk insert into database
            db.bulk_save_objects(stock_data_models)
            db.commit()
            
            logger.info(f"Successfully stored {len(stock_data_models)} data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing data in database: {str(e)}")
            db.rollback()
            raise
    
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None, db: Session = None) -> List[Dict[str, Any]]:
        """
        Get stock data from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data range
            end_date: End date for data range
            db: Database session
            
        Returns:
            List of stock data dictionaries
        """
        try:
            logger.info(f"Getting stock data for {symbol} from database")
            
            if not db:
                logger.warning("No database session provided, returning empty list")
                return []
            
            # Build query
            query = db.query(StockDataModel).filter(StockDataModel.symbol == symbol)
            
            # Apply date filters if provided
            if start_date:
                query = query.filter(StockDataModel.date >= datetime.strptime(start_date, "%Y-%m-%d"))
            if end_date:
                query = query.filter(StockDataModel.date <= datetime.strptime(end_date, "%Y-%m-%d"))
            
            # Execute query
            results = query.order_by(StockDataModel.date).all()
            
            # Convert to list of dictionaries
            stock_data = []
            for item in results:
                stock_data.append({
                    "Date": item.date.strftime("%Y-%m-%d"),
                    "Open": item.open_price,
                    "High": item.high_price,
                    "Low": item.low_price,
                    "Close": item.close_price,
                    "Volume": item.volume
                })
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock data from database: {str(e)}")
            raise
    
    def get_available_symbols(self, db: Session = None) -> List[str]:
        """
        Get list of available stock symbols in the database.
        
        Args:
            db: Database session
            
        Returns:
            List of available stock symbols
        """
        try:
            logger.info("Getting available stock symbols from database")
            
            if not db:
                logger.warning("No database session provided, returning empty list")
                return []
            
            # Execute query
            results = db.query(StockDataModel.symbol).distinct().all()
            
            # Extract symbols from results
            symbols = [result.symbol for result in results]
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            raise
    
    def delete_stock_data(self, symbol: str, db: Session = None) -> Dict[str, Any]:
        """
        Delete stock data for a symbol from the database.
        
        Args:
            symbol: Stock symbol
            db: Database session
            
        Returns:
            Dictionary containing deletion result
        """
        try:
            logger.info(f"Deleting stock data for {symbol} from database")
            
            if not db:
                logger.warning("No database session provided, cannot delete data")
                return {"status": "error", "message": "No database session provided"}
            
            # Execute deletion
            deleted_count = db.query(StockDataModel).filter(StockDataModel.symbol == symbol).delete()
            db.commit()
            
            return {
                "status": "success",
                "message": f"Deleted {deleted_count} records for {symbol}",
                "deleted_count": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting stock data: {str(e)}")
            db.rollback()
            raise

# Singleton instance
_data_service = None

def get_data_service() -> DataService:
    """Get singleton instance of DataService."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service
