from typing import Any, Dict
import os
from datetime import datetime

import pandas as pd

from .base_agent import BaseAgent
from services.data_service import get_data_service
from utils.data_fetcher import fetch_stock_data
from utils.logger import get_logger

logger = get_logger(__name__)


class DataEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Data Engineer",
            description="Load local price data and local precomputed sentiment csv data"
        )
        self.data_service = get_data_service()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            time_range = input_data.get("time_range", "1y")
            interval = input_data.get("interval", "1d")
            db = input_data.get("db")

            sentiment_files = input_data.get("sentiment_files") or self._resolve_default_sentiment_files()
            crawled_at = datetime.now().isoformat()

            logger.info(
                f"DataEngineerAgent loading local price data and local sentiment csvs, "
                f"time_range={time_range}, interval={interval}"
            )

            data_by_symbol = {}
            flat_data = []

            for symbol in ["CSI100", "CSI300"]:
                df = fetch_stock_data(
                    symbol=symbol,
                    interval=interval,
                    start_date="2013-01-01",
                    end_date="2025-01-07"
                )

                if df.empty:
                    logger.warning(f"No price data found for {symbol}")
                    data_by_symbol[symbol] = []
                    continue

                symbol_data = df.reset_index().to_dict(orient="records")
                normalized_records = []

                for item in symbol_data:
                    record = dict(item)
                    if "Date" in record:
                        record["Date"] = pd.to_datetime(record["Date"]).strftime("%Y-%m-%d")
                    record["Symbol"] = symbol
                    normalized_records.append(record)

                data_by_symbol[symbol] = normalized_records
                flat_data.extend(normalized_records)

            sentiment_data_by_symbol = self._load_all_sentiment_data(sentiment_files)

            result = {
                "metadata": {
                    "mode": "local_precomputed_sentiment_csv",
                    "time_range": time_range,
                    "interval": interval,
                    "crawled_at": crawled_at,
                    "symbols": ["CSI100", "CSI300"],
                    "data_points": len(flat_data),
                    "sentiment_files": sentiment_files
                },
                "data": flat_data,
                "data_by_symbol": data_by_symbol,
                "sentiment_data_by_symbol": sentiment_data_by_symbol,
                "news": []
            }

            return self._format_output(
                status="success",
                result=result,
                message="Successfully loaded price data and local sentiment csv data"
            )

        except Exception as e:
            logger.error(f"Error in DataEngineerAgent: {str(e)}")
            return self._format_output(
                status="error",
                result=None,
                message=str(e)
            )

    def _resolve_default_sentiment_files(self) -> Dict[str, str]:
        _documents_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "documents"))
        candidates = {
            "CSI100": [
                os.path.join(_documents_dir, "CSI100.csv"),
                os.path.join(os.getcwd(), "CSI100.csv"),
                os.path.join(os.getcwd(), "data", "CSI100.csv"),
            ],
            "CSI300": [
                os.path.join(_documents_dir, "CSI300.csv"),
                os.path.join(os.getcwd(), "CSI300.csv"),
                os.path.join(os.getcwd(), "data", "CSI300.csv"),
            ],
        }

        resolved = {}
        for symbol, paths in candidates.items():
            for path in paths:
                if os.path.exists(path):
                    resolved[symbol] = path
                    break

        if set(resolved.keys()) != {"CSI100", "CSI300"}:
            raise FileNotFoundError(
                "Could not resolve local sentiment csv files for CSI100 and CSI300. "
                "Please pass input_data['sentiment_files'] explicitly."
            )

        return resolved

    def _load_sentiment_csv(self, file_path: str, symbol: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sentiment file not found for {symbol}: {file_path}")

        df = pd.read_csv(file_path)
        if "trade_date" not in df.columns:
            raise ValueError(f"{symbol} sentiment file must contain 'trade_date' column")

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df = df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
        df["Symbol"] = symbol

        records = df.to_dict(orient="records")
        normalized_records = []
        for row in records:
            row = dict(row)
            row["trade_date"] = pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d")
            normalized_records.append(row)

        logger.info(f"Loaded {len(normalized_records)} sentiment rows for {symbol} from {file_path}")
        return normalized_records

    def _load_all_sentiment_data(self, sentiment_files: Dict[str, str]) -> Dict[str, Any]:
        results = {}
        for symbol in ["CSI100", "CSI300"]:
            if symbol not in sentiment_files:
                raise ValueError(f"Missing sentiment file path for {symbol}")
            results[symbol] = self._load_sentiment_csv(sentiment_files[symbol], symbol)
        return results

    def _clean_crawled_data(self, data):
        cleaned = [item for item in data if item and all(value is not None for value in item.values())]
        if cleaned and "Date" in cleaned[0]:
            cleaned.sort(key=lambda x: x["Date"])
        return cleaned