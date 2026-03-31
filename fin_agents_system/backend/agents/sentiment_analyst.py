from datetime import datetime
from typing import Any, Dict

import pandas as pd

from .base_agent import BaseAgent
from services.llm_service import get_llm_service
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Sentiment Analyst",
            description="Load precomputed sentiment features or fallback to news sentiment analysis"
        )
        self.llm_service = get_llm_service()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "sentiment_data_by_symbol" in input_data:
                result = self._load_precomputed_sentiment(input_data["sentiment_data_by_symbol"])
                return self._format_output(
                    status="success",
                    result=result,
                    message="Successfully loaded precomputed sentiment features"
                )

            if "news" in input_data:
                news = input_data["news"]
                timeframe = input_data.get("timeframe", "recent")

                sentiment_results = self._analyze_sentiment(news)
                if not sentiment_results:
                    sentiment_results = [{
                        "content": "No news data available for analysis",
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "timestamp": datetime.now().isoformat(),
                        "source_url": "unknown",
                        "analysis_time": datetime.now().isoformat()
                    }]

                sentiment_by_date = self._group_sentiment_by_date(sentiment_results)
                overall_score = self._calculate_overall_score(sentiment_results)
                summary = self._generate_summary(sentiment_results, overall_score)
                sentiment_dimensions = self._calculate_sentiment_dimensions(sentiment_by_date)

                result = {
                    "timeframe": timeframe,
                    "sentiment_score": overall_score,
                    "sentiment_classification": self._classify_sentiment(overall_score),
                    "sentiment_breakdown": sentiment_results,
                    "sentiment_by_date": sentiment_by_date,
                    "sentiment_dimensions": sentiment_dimensions,
                    "summary": summary
                }

                return self._format_output(
                    status="success",
                    result=result,
                    message="Successfully analyzed sentiment from news"
                )

            return self._format_output(
                status="error",
                result=None,
                message="Missing 'sentiment_data_by_symbol' or 'news'"
            )

        except Exception as e:
            logger.error(f"Error in SentimentAnalystAgent: {str(e)}")
            return self._format_output(
                status="error",
                result=None,
                message=str(e)
            )

    def _load_precomputed_sentiment(self, sentiment_data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        by_symbol = {}

        for symbol, records in sentiment_data_by_symbol.items():
            df = pd.DataFrame(records)
            if df.empty:
                by_symbol[symbol] = {
                    "timeframe": "historical",
                    "sentiment_score": 0.0,
                    "sentiment_classification": "neutral",
                    "sentiment_breakdown": [],
                    "sentiment_by_date": {},
                    "sentiment_dimensions": {},
                    "summary": f"No precomputed sentiment data for {symbol}",
                    "source": "precomputed_csv"
                }
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
            df = df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

            if "Symbol" in df.columns:
                df = df.drop(columns=["Symbol"])

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            sentiment_dimensions = {}
            sentiment_by_date = {}

            for _, row in df.iterrows():
                date_key = row["trade_date"].strftime("%Y-%m-%d")
                row_dict = row.to_dict()
                row_dict["trade_date"] = date_key
                sentiment_dimensions[date_key] = row_dict

                day_score = float(
                    row_dict.get(
                        "combined_sentiment",
                        row_dict.get("quality_tone", row_dict.get("avg_tone", 0.0))
                    )
                )

                sentiment_by_date[date_key] = {
                    "date": date_key,
                    "average_score": round(day_score, 6),
                    "news_count": int(row_dict.get("total_news", 0)),
                    "dominant_sentiment": self._classify_precomputed_score(day_score)
                }

            overall_score = round(
                float(df["combined_sentiment"].mean()) if "combined_sentiment" in df.columns else 0.0,
                6
            )

            by_symbol[symbol] = {
                "timeframe": "historical",
                "sentiment_score": overall_score,
                "sentiment_classification": self._classify_precomputed_score(overall_score),
                "sentiment_breakdown": [],
                "sentiment_by_date": sentiment_by_date,
                "sentiment_dimensions": sentiment_dimensions,
                "summary": (
                    f"{symbol} precomputed sentiment loaded from local csv, "
                    f"{len(df)} aligned daily rows, average combined_sentiment={overall_score:.6f}"
                ),
                "source": "precomputed_csv",
                "feature_columns": numeric_cols
            }

        all_scores = [
            v.get("sentiment_score", 0.0)
            for v in by_symbol.values()
        ]
        overall = round(sum(all_scores) / len(all_scores), 6) if all_scores else 0.0

        return {
            "timeframe": "historical",
            "source": "precomputed_csv",
            "sentiment_score": overall,
            "sentiment_classification": self._classify_precomputed_score(overall),
            "by_symbol": by_symbol,
            "summary": "Loaded precomputed sentiment features for CSI100 and CSI300"
        }

    def _classify_precomputed_score(self, score: float) -> str:
        if score >= 0.55:
            return "positive"
        if score <= 0.45:
            return "negative"
        return "neutral"

    def _analyze_sentiment(self, news: list) -> list:
        results = []

        for news_item in news:
            if isinstance(news_item, dict):
                content = news_item.get("content", "")
                timestamp = news_item.get("publication_date", news_item.get("date", datetime.now().isoformat()))
                source_url = news_item.get("source_url", "unknown")
            else:
                content = news_item
                timestamp = datetime.now().isoformat()
                source_url = "unknown"

            if not content:
                continue

            sentiment = self.llm_service.analyze_sentiment(content)
            sentiment_with_metadata = {
                "content": content,
                "sentiment_score": sentiment.get("score", 0),
                "sentiment_label": sentiment.get("label", sentiment.get("sentiment", "neutral")),
                "timestamp": timestamp,
                "source_url": source_url,
                "analysis_time": datetime.now().isoformat()
            }
            results.append(sentiment_with_metadata)

        return results

    def _group_sentiment_by_date(self, sentiment_results: list) -> Dict[str, Dict[str, Any]]:
        sentiment_by_date = {}

        for result in sentiment_results:
            try:
                timestamp = result.get("timestamp", datetime.now().isoformat())
                date_obj = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
                date_key = date_obj.strftime("%Y-%m-%d")

                if date_key not in sentiment_by_date:
                    sentiment_by_date[date_key] = {
                        "date": date_key,
                        "sentiment_scores": [],
                        "sentiment_labels": [],
                        "news_count": 0,
                        "average_score": 0,
                        "dominant_sentiment": "neutral"
                    }

                sentiment_by_date[date_key]["sentiment_scores"].append(result["sentiment_score"])
                sentiment_by_date[date_key]["sentiment_labels"].append(result["sentiment_label"])
                sentiment_by_date[date_key]["news_count"] += 1

            except Exception as e:
                logger.warning(f"Error processing sentiment timestamp: {str(e)}")
                continue

        for date_key, data in sentiment_by_date.items():
            scores = data["sentiment_scores"]
            if scores:
                average_score = sum(scores) / len(scores)
                sentiment_by_date[date_key]["average_score"] = round(average_score, 4)

                from collections import Counter
                label_counts = Counter(data["sentiment_labels"])
                sentiment_by_date[date_key]["dominant_sentiment"] = label_counts.most_common(1)[0][0]

        return sentiment_by_date

    def _calculate_overall_score(self, sentiment_results: list) -> float:
        if not sentiment_results:
            return 0.0
        total_score = sum(result.get("sentiment_score", 0) for result in sentiment_results)
        return round(total_score / len(sentiment_results), 2)

    def _calculate_sentiment_dimensions(self, sentiment_by_date: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        import numpy as np

        sorted_dates = sorted(sentiment_by_date.keys())
        dates_data = []

        for date in sorted_dates:
            data = sentiment_by_date[date]
            dates_data.append({
                "date": date,
                "news_count": data["news_count"],
                "avg_tone": data["average_score"],
                "positive_count": data["sentiment_labels"].count("positive"),
                "neutral_count": data["sentiment_labels"].count("neutral"),
                "negative_count": data["sentiment_labels"].count("negative")
            })

        result = {}
        for i, date_data in enumerate(dates_data):
            date = date_data["date"]
            total_news = date_data["news_count"]
            avg_tone = date_data["avg_tone"]

            positive_ratio = date_data["positive_count"] / total_news if total_news > 0 else 0.0

            historical_news = [d["news_count"] for d in dates_data[max(0, i-20):i+1]]
            historical_tone = [d["avg_tone"] for d in dates_data[max(0, i-20):i+1]]
            historical_pos_ratio = [
                d["positive_count"] / d["news_count"] if d["news_count"] > 0 else 0.0
                for d in dates_data[max(0, i-20):i+1]
            ]

            news_ma_5 = np.mean(historical_news[-5:]) if len(historical_news) >= 5 else historical_news[-1] if historical_news else 0.0
            news_std_5 = np.std(historical_news[-5:]) if len(historical_news) >= 5 else 0.0
            news_ma_10 = np.mean(historical_news[-10:]) if len(historical_news) >= 10 else historical_news[-1] if historical_news else 0.0
            news_std_10 = np.std(historical_news[-10:]) if len(historical_news) >= 10 else 0.0
            news_ma_20 = np.mean(historical_news[-20:]) if len(historical_news) >= 20 else historical_news[-1] if historical_news else 0.0
            news_std_20 = np.std(historical_news[-20:]) if len(historical_news) >= 20 else 0.0

            news_zscore_5 = (historical_news[-1] - news_ma_5) / (news_std_5 + 1e-8) if len(historical_news) >= 5 else 0.0
            news_zscore_10 = (historical_news[-1] - news_ma_10) / (news_std_10 + 1e-8) if len(historical_news) >= 10 else 0.0
            news_zscore_20 = (historical_news[-1] - news_ma_20) / (news_std_20 + 1e-8) if len(historical_news) >= 20 else 0.0

            news_daily_change = historical_news[-1] - historical_news[-2] if len(historical_news) >= 2 else 0.0
            news_weekly_change = historical_news[-1] - historical_news[-7] if len(historical_news) >= 7 else 0.0

            tone_ma_3 = np.mean(historical_tone[-3:]) if len(historical_tone) >= 3 else historical_tone[-1] if historical_tone else 0.0
            tone_volatility_3 = np.std(historical_tone[-3:]) if len(historical_tone) >= 3 else 0.0
            tone_ma_5 = np.mean(historical_tone[-5:]) if len(historical_tone) >= 5 else historical_tone[-1] if historical_tone else 0.0
            tone_volatility_5 = np.std(historical_tone[-5:]) if len(historical_tone) >= 5 else 0.0
            tone_ma_10 = np.mean(historical_tone[-10:]) if len(historical_tone) >= 10 else historical_tone[-1] if historical_tone else 0.0
            tone_volatility_10 = np.std(historical_tone[-10:]) if len(historical_tone) >= 10 else 0.0

            tone_change = historical_tone[-1] - historical_tone[-2] if len(historical_tone) >= 2 else 0.0
            tone_momentum = sum(historical_tone[max(0, len(historical_tone)-5):]) if historical_tone else 0.0

            pos_ratio_ma_5 = np.mean(historical_pos_ratio[-5:]) if len(historical_pos_ratio) >= 5 else historical_pos_ratio[-1] if historical_pos_ratio else 0.0
            pos_ratio_ma_10 = np.mean(historical_pos_ratio[-10:]) if len(historical_pos_ratio) >= 10 else historical_pos_ratio[-1] if historical_pos_ratio else 0.0
            pos_ratio_ma_20 = np.mean(historical_pos_ratio[-20:]) if len(historical_pos_ratio) >= 20 else historical_pos_ratio[-1] if historical_pos_ratio else 0.0

            pos_ratio_change = historical_pos_ratio[-1] - historical_pos_ratio[-2] if len(historical_pos_ratio) >= 2 else 0.0
            pos_ratio_acc = (historical_pos_ratio[-1] - 2 * historical_pos_ratio[-2] + historical_pos_ratio[-3]) if len(historical_pos_ratio) >= 3 else 0.0

            weighted_tone = avg_tone * total_news / (total_news + 1e-8)
            quality_tone = avg_tone
            combined_sentiment = (weighted_tone + positive_ratio) / 2

            result[date] = {
                "trade_date": date,
                "total_news": total_news,
                "avg_tone": avg_tone,
                "positive_ratio": positive_ratio,
                "news_ma_5": news_ma_5,
                "news_std_5": news_std_5,
                "news_zscore_5": news_zscore_5,
                "news_ma_10": news_ma_10,
                "news_std_10": news_std_10,
                "news_zscore_10": news_zscore_10,
                "news_ma_20": news_ma_20,
                "news_std_20": news_std_20,
                "news_zscore_20": news_zscore_20,
                "news_daily_change": news_daily_change,
                "news_weekly_change": news_weekly_change,
                "tone_ma_3": tone_ma_3,
                "tone_volatility_3": tone_volatility_3,
                "tone_ma_5": tone_ma_5,
                "tone_volatility_5": tone_volatility_5,
                "tone_ma_10": tone_ma_10,
                "tone_volatility_10": tone_volatility_10,
                "tone_change": tone_change,
                "tone_momentum": tone_momentum,
                "pos_ratio_ma_5": pos_ratio_ma_5,
                "pos_ratio_ma_10": pos_ratio_ma_10,
                "pos_ratio_ma_20": pos_ratio_ma_20,
                "pos_ratio_change": pos_ratio_change,
                "pos_ratio_acc": pos_ratio_acc,
                "weighted_tone": weighted_tone,
                "quality_tone": quality_tone,
                "combined_sentiment": combined_sentiment
            }

        return result

    def _classify_sentiment(self, score: float) -> str:
        if score > 0.3:
            return "positive"
        elif score < -0.3:
            return "negative"
        else:
            return "neutral"

    def _generate_summary(self, sentiment_results: list, overall_score: float) -> str:
        return self.llm_service.generate_sentiment_summary(sentiment_results, overall_score)