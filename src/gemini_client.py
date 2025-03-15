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

    def get_trading_decision(self, market_data: list[float], lookback_period: int = 20):
        """
        Optimized method to generate a trading decision using advanced price action analysis.
        
        Args:
            market_data (list[float]): List of closing prices (5-minute candles).
            lookback_period (int): Number of candles to consider for volatility and trend calculations.

        Returns:
            dict: Trading decision with entry price, stop-loss, take-profit, trend, and trend strength.
        """
        # Calculate recent price action context
        recent_prices = market_data[-lookback_period:]  # Last N candles for context
        price_change = recent_prices[-1] - recent_prices[0]
        trend = "uptrend" if price_change > 0 else "downtrend" if price_change < 0 else "sideways"

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(recent_prices)

        # Calculate dynamic volatility
        volatility = self._calculate_volatility(recent_prices)

        prompt = f"""
        You are a **god-level crypto scalper** with unparalleled expertise in 5-minute timeframe trading. 
        Your task is to analyze the provided price action data and make a trading decision (buy, sell, or hold) based on **advanced price action principles**. 
        Use the following guidelines to determine the optimal entry, stop-loss, and take-profit levels.

        **Market Data:**
        - 5-minute closing prices: {market_data}
        - Recent trend: {trend}
        - Trend strength: {trend_strength}
        - Volatility: {volatility:.4f}

        **Advanced Price Action Analysis:**
        1. **Trend Identification**:
           - Confirm the current trend using higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend).
           - Look for **trend continuation patterns** (e.g., flags, pennants) or **trend reversal patterns** (e.g., double tops/bottoms).

        2. **Key Support and Resistance Levels**:
           - Identify major support and resistance levels based on recent price action.
           - Look for **breakouts** or **breakdowns** of these levels with strong momentum.
           - Example: If price breaks above resistance with high volume, consider a buy signal.

        3. **Candlestick Patterns**:
           - Analyze candlestick patterns for confirmation:
             - **Bullish Patterns**: Hammer, bullish engulfing, morning star.
             - **Bearish Patterns**: Shooting star, bearish engulfing, evening star.
           - Look for **wick rejection** at key levels for additional confirmation.

        4. **Volume Analysis**:
           - Confirm breakouts/breakdowns with **increased volume**.
           - Low volume during consolidation indicates indecision.

        5. **Risk Management**:
           - Use the calculated **volatility ({volatility:.4f})** to dynamically adjust stop-loss and take-profit.
           - **Take-profit = Entry price ± (volatility * 3)** (wider take-profit for better risk-reward).
           - **Stop-loss = Entry price ± (volatility * 1.5)** (wider stop-loss to avoid premature exits).
           - Ensure a **risk-reward ratio of at least 1:2**.

        **Trading Decision Rules:**
        1. **Buy Signal**:
           - Price breaks above a key resistance level with strong bullish momentum.
           - Bullish candlestick patterns (e.g., hammer, bullish engulfing) form near support.
           - Volume increases significantly during the breakout.
           - Set stop-loss below the recent swing low and take-profit at the next resistance level.

        2. **Sell Signal**:
           - Price breaks below a key support level with strong bearish momentum.
           - Bearish candlestick patterns (e.g., shooting star, bearish engulfing) form near resistance.
           - Volume increases significantly during the breakdown.
           - Set stop-loss above the recent swing high and take-profit at the next support level.

        3. **Hold Signal**:
           - Price is consolidating between support and resistance levels.
           - No clear breakout or breakdown is observed.
           - Candlestick patterns show indecision (e.g., doji).

        **Return JSON Format:**
        {{
            "decision": "buy/sell/hold",
            "reason": "Detailed explanation of the decision based on price action analysis",
            "entry_price": "Optimal entry price based on current market conditions",
            "stop_loss": "Calculated stop-loss level based on volatility and price action",
            "take_profit": "Calculated take-profit level based on volatility and price action",
            "trend": "{trend}",
            "trend_strength": "{trend_strength}"
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
                "decision": "hold",
                "reason": "Unable to parse response",
                "entry_price": market_data[-1],
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

        if decision_data["decision"] not in ["buy", "sell", "hold"]:
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