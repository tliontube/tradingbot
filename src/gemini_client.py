import google.generativeai as genai
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def get_trading_decision(self, closing_prices: list[float], volumes: list[float], 
                          high_prices: list[float], low_prices: list[float], 
                          open_prices: list[float], lookback_period: int = 100):
        """
        Optimized method to generate a trading decision using advanced price action analysis.
        Incorporates volume analysis for better confirmation of breakouts and trends.
        
        Args:
            closing_prices (list[float]): List of closing prices
            volumes (list[float]): List of trading volumes
            high_prices (list[float]): List of high prices
            low_prices (list[float]): List of low prices
            open_prices (list[float]): List of open prices
            lookback_period (int): Number of candles to consider

        Returns:
            dict: Trading decision with entry price, stop-loss, take-profit, trend, and trend strength.
        """
        # Calculate recent price action context
        recent_prices = closing_prices[-lookback_period:]
        recent_volumes = volumes[-lookback_period:]
        recent_highs = high_prices[-lookback_period:]
        recent_lows = low_prices[-lookback_period:]
        
        price_change = recent_prices[-1] - recent_prices[0]
        trend = "uptrend" if price_change > 0 else "downtrend" if price_change < 0 else "sideways"

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(recent_prices)

        # Calculate dynamic volatility
        volatility = self._calculate_volatility(recent_prices)

        # Calculate volume metrics
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        recent_volume = sum(recent_volumes[-3:]) / 3  # Average of last 3 candles
        volume_increase = recent_volume / avg_volume if avg_volume > 0 else 1.0

        prompt = f"""
        You are a **professional crypto trader** with expertise in price action and volume analysis. Your task is to analyze the provided market data and make a trading decision based on **advanced price action principles** and **volume analysis**. 
        
        IMPORTANT: Only generate buy/sell signals when there is:
        1. A clear breakout with confirmation (price breaking key levels with volume)
        2. A strong trend continuation pattern in a moderate/strong trend
        3. Supporting volume evidence (increased volume on breakouts, decreased volume on consolidations)
        Otherwise, provide a detailed market analysis explaining why we should wait.

        **Market Data:**
        - 5-minute closing prices: {recent_prices}
        - Trading volumes: {recent_volumes}
        - High prices: {recent_highs}
        - Low prices: {recent_lows}
        - Recent trend: {trend}
        - Trend strength: {trend_strength}
        - Volatility: {volatility:.4f}
        - Volume Analysis:
          * Average Volume: {avg_volume:.2f}
          * Recent Volume: {recent_volume:.2f}
          * Volume Change: {(volume_increase - 1) * 100:.1f}% from average

        **Advanced Price Action Analysis:**
        1. **Volume-Confirmed Breakouts**:
           - Look for clear breaks of key support/resistance levels
           - Require volume increase of at least 50% above average on breakouts
           - Multiple candle confirmation of the break
           - Avoid breakouts with declining volume

        2. **Volume-Trend Analysis**:
           - Check if volume supports the price trend
           - Higher volume on trending moves
           - Lower volume on retracements
           - Volume should increase in trend direction

        3. **Risk Management**:
           - Use the calculated **volatility ({volatility:.4f})** to dynamically adjust stop-loss and take-profit
           - **Take-profit = Entry price ± (volatility * 3)**
           - **Stop-loss = Entry price ± (volatility * 1.5)**
           - Ensure a **risk-reward ratio of at least 1:2**

        **Trading Decision Rules:**
        1. **Buy Signal** (ALL conditions must be met):
           - Clear breakout above key resistance with confirmation
           - Strong bullish momentum in moderate/strong uptrend
           - Volume increase >50% above average on breakout
           - Multiple candle confirmation
           - Higher highs and higher lows forming

        2. **Sell Signal** (ALL conditions must be met):
           - Clear breakdown below key support with confirmation
           - Strong bearish momentum in moderate/strong downtrend
           - Volume increase >50% above average on breakdown
           - Multiple candle confirmation
           - Lower highs and lower lows forming

        3. **Waiting Signal**:
           - Provide a detailed analysis of current market conditions
           - Explain specifically why no trade should be taken
           - Analyze both price action AND volume patterns
           - Identify what conditions are missing for a trade
           - Suggest what conditions to watch for

        **Return JSON Format:**
        {{
            "decision": "buy/sell/waiting",
            "reason": "Detailed market analysis explaining the decision. For waiting decisions, provide specific reasons why no trade should be taken and what conditions to watch for. Include current price, key support/resistance levels, and volume analysis.",
            "entry_price": "Optimal entry price based on current market conditions",
            "stop_loss": "Calculated stop-loss level based on volatility and price action",
            "take_profit": "Calculated take-profit level based on volatility and price action",
            "trend": "{trend}",
            "trend_strength": "{trend_strength}",
            "volume_analysis": {{
                "average_volume": {avg_volume:.2f},
                "recent_volume": {recent_volume:.2f},
                "volume_change": "{(volume_increase - 1) * 100:.1f}%"
            }}
        }}
        """

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            response_text = response.text.replace("```json", "").replace("```", "").strip()
            decision_data = json.loads(response_text)

            # Add trend and trend strength to the decision data
            decision_data["trend"] = trend
            decision_data["trend_strength"] = trend_strength

            # Validate the decision data
            if not self._validate_decision(decision_data):
                raise ValueError("Invalid decision data from Gemini")

            logger.info(f"Gemini Decision: {decision_data}")
            return decision_data

        except Exception as e:
            logger.error(f"Error in GeminiClient: {e}")
            return {
                "decision": "waiting",
                "reason": f"Error processing market data: {str(e)}. Please check the system logs for more details.",
                "entry_price": recent_prices[-1],
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "trend_strength": trend_strength
            }

    def _validate_decision(self, decision_data: dict) -> bool:
        """
        Validate the decision data returned by Gemini.

        Args:
            decision_data (dict): Decision data to validate.

        Returns:
            bool: True if the decision data is valid, False otherwise.
        """
        required_keys = ["decision", "reason", "entry_price", "stop_loss", "take_profit", "trend", "trend_strength"]
        if not all(key in decision_data for key in required_keys):
            return False

        if decision_data["decision"] not in ["buy", "sell", "waiting"]:
            return False

        if decision_data["trend"] not in ["uptrend", "downtrend", "sideways"]:
            return False

        if decision_data["trend_strength"] not in ["weak", "moderate", "strong"]:
            return False

        return True

    def _calculate_trend_strength(self, prices: list[float]) -> str:
        """
        Calculate the strength of the trend based on recent price action.

        Args:
            prices (list[float]): List of recent closing prices.

        Returns:
            str: Trend strength (weak, moderate, or strong).
        """
        price_change = prices[-1] - prices[0]
        absolute_change = abs(price_change)
        avg_price = sum(prices) / len(prices)
        change_percentage = (absolute_change / avg_price) * 100

        if change_percentage < 1:
            return "weak"
        elif 1 <= change_percentage < 3:
            return "moderate"
        else:
            return "strong"

    def _calculate_volatility(self, prices: list[float]) -> float:
        """
        Calculate dynamic volatility based on recent price movements.

        Args:
            prices (list[float]): List of recent closing prices.

        Returns:
            float: Volatility factor.
        """
        returns = np.diff(prices) / prices[:-1]  # Calculate daily returns
        volatility = np.std(returns)  # Standard deviation of returns
        return volatility