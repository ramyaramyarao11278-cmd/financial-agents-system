from typing import Any, Dict, List

from config import config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.logger import get_logger

# Get logger first
logger = get_logger(__name__)

# Only use ChatOpenAI, remove Ollama support
HAS_OPENAI = True

class LLMService:
    """Service for interacting with large language models."""
    
    def __init__(self):
        self.model = config.LLM_MODEL
        self.openai_api_key = config.OPENAI_API_KEY
        self.openai_base_url = config.OPENAI_BASE_URL
        self.use_mock = False
        self.chat_model = None
        
        # Initialize LLM model with ChatOpenAI only
        try:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self.chat_model = ChatOpenAI(
                model=self.model,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            logger.info(f"Initialized LLMService with ChatOpenAI model: {self.model}")
            logger.info(f"Using OpenAI base URL: {self.openai_base_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOpenAI, falling back to mock implementation: {e}")
            self.use_mock = True
        
        logger.info(f"LLMService initialized with use_mock: {self.use_mock}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a given text using LLM.
        
        Args:
            text: Text to analyze sentiment for.
            
        Returns:
            Dictionary containing sentiment analysis results.
        """
        try:
            logger.info(f"Analyzing sentiment for text: {text[:50]}...")
            
            # Use mock implementation if configured
            if self.use_mock:
                logger.info("Using mock sentiment analysis")
                # Simple mock implementation based on keyword matching
                positive_words = ["strong", "beating", "upgrade", "boost", "success", "growth", "profit", "exceed"]
                negative_words = ["volatility", "decline", "loss", "downgrade", "risk", "warning", "fail", "drop"]
                
                text_lower = text.lower()
                score = 0.0
                
                # Count positive and negative words
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                # Calculate simple sentiment score
                if positive_count > negative_count:
                    score = 0.5 * positive_count / max(positive_count + negative_count, 1)
                elif negative_count > positive_count:
                    score = -0.5 * negative_count / max(positive_count + negative_count, 1)
                else:
                    score = 0.0
                
                # Determine sentiment category
                if score > 0.2:
                    sentiment = "positive"
                elif score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                return {
                    "text": text,
                    "score": round(score, 2),
                    "sentiment": sentiment,
                    "confidence": 0.7  # Mock confidence
                }
            
            # Create prompt template for sentiment analysis if using real LLM
            prompt = ChatPromptTemplate.from_template("""Analyze the sentiment of the following text:
            
            Text: {text}
            
            Please provide the sentiment analysis in the following JSON format:
            {{
                "text": "{text}",
                "score": <float between -1 and 1>,
                "sentiment": <"positive", "negative", or "neutral">,
                "confidence": <float between 0 and 1>
            }}
            
            Where:
            - score: -1 is very negative, 1 is very positive
            - sentiment: the overall sentiment category
            - confidence: how confident you are in the analysis
            """)
            
            # Create chain
            chain = prompt | self.chat_model | StrOutputParser()
            
            # Get response from LLM
            response = chain.invoke({"text": text})
            
            # Parse response
            import json
            result = json.loads(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            # Return default neutral sentiment in case of error
            return {
                "text": text,
                "score": 0.0,
                "sentiment": "neutral",
                "confidence": 0.5
            }
    
    def generate_sentiment_summary(self, sentiment_results: List[Dict[str, Any]], overall_score: float) -> str:
        """
        Generate a summary of sentiment analysis results using LLM.
        
        Args:
            sentiment_results: List of sentiment analysis results.
            overall_score: Overall sentiment score.
            
        Returns:
            Summary of sentiment analysis.
        """
        try:
            logger.info("Generating sentiment summary")
            
            # Use mock implementation if configured
            if self.use_mock:
                logger.info("Using mock sentiment summary generation")
                
                # Calculate sentiment breakdown
                positive_count = sum(1 for r in sentiment_results if r["sentiment"] == "positive")
                negative_count = sum(1 for r in sentiment_results if r["sentiment"] == "negative")
                neutral_count = sum(1 for r in sentiment_results if r["sentiment"] == "neutral")
                total_count = len(sentiment_results)
                
                # Determine overall sentiment
                if overall_score > 0.3:
                    overall_sentiment = "Positive"
                elif overall_score < -0.3:
                    overall_sentiment = "Negative"
                else:
                    overall_sentiment = "Neutral"
                
                # Generate simple mock summary
                summary = f"{overall_sentiment} sentiment with an overall score of {overall_score:.2f}. "
                summary += f"Out of {total_count} articles, {positive_count} were positive, {negative_count} were negative, and {neutral_count} were neutral. "
                
                if overall_sentiment == "Positive":
                    summary += "The positive sentiment suggests favorable market conditions and potential growth opportunities."
                elif overall_sentiment == "Negative":
                    summary += "The negative sentiment indicates caution and potential market volatility ahead."
                else:
                    summary += "The neutral sentiment reflects a balanced market outlook with no strong directional bias."
                
                return summary
            
            # Create prompt template for sentiment summary if using real LLM
            prompt = ChatPromptTemplate.from_template("""Generate a concise summary of the following sentiment analysis results:
            
            Sentiment Results: {sentiment_results}
            Overall Score: {overall_score}
            
            The summary should:
            1. Start with the overall sentiment classification
            2. Mention the overall score
            3. Include the breakdown of positive, negative, and neutral articles
            4. Provide insights about what the sentiment means for the market
            5. Be professional and concise
            """)
            
            # Create chain
            chain = prompt | self.chat_model | StrOutputParser()
            
            # Get response from LLM
            response = chain.invoke({
                "sentiment_results": sentiment_results,
                "overall_score": overall_score
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {str(e)}")
            # Return simple mock summary in case of error
            return f"Sentiment summary generation failed. Overall sentiment score: {overall_score:.2f}"
    
    def generate_analysis_report(self, data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report based on multiple data sources using LLM.
        
        Args:
            data: Dictionary containing various analysis results.
            
        Returns:
            Comprehensive analysis report.
        """
        try:
            logger.info("Generating comprehensive analysis report")
            
            # Use mock implementation if configured
            if self.use_mock:
                logger.info("Using mock analysis report generation")
                
                # Generate simple mock report
                report = "# Financial Analysis Report\n\n"
                report += "## Executive Summary\n"
                report += "This is a mock analysis report generated using the mock LLM implementation.\n\n"
                
                report += "## Market Sentiment Analysis\n"
                report += "The market sentiment is currently neutral.\n\n"
                
                report += "## Technical Trend Analysis\n"
                report += "The technical indicators suggest a neutral trend.\n\n"
                
                report += "## Key Insights\n"
                report += "- Mock analysis shows no significant trends\n"
                report += "- Market conditions appear stable\n\n"
                
                report += "## Trading Recommendations\n"
                report += "Hold position. Further analysis required.\n\n"
                
                report += "## Risk Assessment\n"
                report += "Low to moderate risk based on current market conditions.\n"
                
                return report
            
            # Create prompt template for analysis report if using real LLM
            prompt = ChatPromptTemplate.from_template("""Generate a comprehensive analysis report based on the following data:
            
            {data}
            
            The report should be structured with clear sections including:
            1. Executive Summary
            2. Market Sentiment Analysis
            3. Technical Trend Analysis
            4. Key Insights
            5. Trading Recommendations
            6. Risk Assessment
            
            The report should be professional, data-driven, and provide actionable insights for traders.
            Use Markdown formatting for the report.
            """)
            
            # Create chain
            chain = prompt | self.chat_model | StrOutputParser()
            
            # Get response from LLM
            response = chain.invoke({"data": data})
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            # Return mock report in case of error
            return "# Financial Analysis Report\n\n## Executive Summary\nAnalysis report generation failed. Please try again later."
    
    def generate_backtest_summary(self, backtest_results: Dict[str, Any]) -> str:
        """
        Generate a summary of backtest results using LLM.
        
        Args:
            backtest_results: Dictionary containing backtest results.
            
        Returns:
            Summary of backtest results.
        """
        try:
            logger.info("Generating backtest summary")
            
            # Use mock implementation if configured
            if self.use_mock:
                logger.info("Using mock backtest summary generation")
                
                # Extract key metrics if available
                total_return = backtest_results.get("total_return", 0)
                annualized_return = backtest_results.get("annualized_return", 0)
                sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
                max_drawdown = backtest_results.get("max_drawdown", 0)
                win_rate = backtest_results.get("win_rate", 0)
                
                # Generate simple mock summary
                summary = f"# Backtest Results Summary\n\n"
                summary += "## Key Performance Metrics\n"
                summary += f"- Total Return: {total_return}%\n"
                summary += f"- Annualized Return: {annualized_return}%\n"
                summary += f"- Sharpe Ratio: {sharpe_ratio}\n"
                summary += f"- Maximum Drawdown: {max_drawdown}%\n"
                summary += f"- Win Rate: {win_rate}%\n\n"
                
                summary += "## Strategy Analysis\n"
                summary += "This is a mock backtest summary generated using the mock LLM implementation.\n\n"
                
                if sharpe_ratio > 1.0:
                    summary += "The strategy demonstrates good risk-adjusted returns.\n"
                else:
                    summary += "The strategy's risk-adjusted returns could be improved.\n"
                
                if win_rate > 50:
                    summary += "The strategy shows a winning edge.\n\n"
                else:
                    summary += "The strategy needs improvement in win rate.\n\n"
                
                summary += "## Recommendations\n"
                summary += "1. Consider optimizing entry and exit signals\n"
                summary += "2. Test the strategy across different market conditions\n"
                summary += "3. Explore parameter tuning to improve performance\n"
                summary += "4. Consider adding additional risk management measures\n"
                
                return summary
            
            # Create prompt template for backtest summary if using real LLM
            prompt = ChatPromptTemplate.from_template("""Generate a detailed and insightful summary of the following backtest results:
            
            {backtest_results}
            
            The summary should:
            1. Highlight key performance metrics
            2. Analyze the strategy's strengths and weaknesses
            3. Provide context for the results
            4. Offer recommendations for improvement if applicable
            5. Be professional and data-driven
            """)
            
            # Create chain
            chain = prompt | self.chat_model | StrOutputParser()
            
            # Get response from LLM
            response = chain.invoke({"backtest_results": backtest_results})
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating backtest summary: {str(e)}")
            # Return simple mock summary in case of error
            return f"# Backtest Results Summary\n\n## Error\nFailed to generate backtest summary: {str(e)}"

# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get singleton instance of LLMService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
