import google.generativeai as genai
import json
import logging
import numpy as np
from machine_learning.trade_analyzer import TradeAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.trade_analyzer = TradeAnalyzer()
        self.base_prompt = """...existing prompt..."""

    def get_trading_decision(self, closing_prices: list[float], volumes: list[float], 
                          high_prices: list[float], low_prices: list[float], 
                          open_prices: list[float], symbol: str = "BTCUSDT", lookback_period: int = 200):
        """
        Advanced price action analysis incorporating Smart Money Concepts (SMC),
        Break of Structure (BOS), and Change of Character (CHoCH).
        
        Args:
            closing_prices (list[float]): List of closing prices
            volumes (list[float]): List of trading volumes (not used in pure price action)
            high_prices (list[float]): List of high prices
            low_prices (list[float]): List of low prices
            open_prices (list[float]): List of open prices
            symbol (str): Trading pair symbol being analyzed
            lookback_period (int): Number of candles to consider

        Returns:
            dict: Trading decision with entry price, stop-loss, take-profit, trend, and trend strength.
        """
        # Calculate recent price action context
        recent_prices = closing_prices[-lookback_period:]
        recent_highs = high_prices[-lookback_period:]
        recent_lows = low_prices[-lookback_period:]
        recent_opens = open_prices[-lookback_period:]
        
        price_change = recent_prices[-1] - recent_prices[0]
        trend = "uptrend" if price_change > 0 else "downtrend" if price_change < 0 else "sideways"

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(recent_prices)

        # Calculate dynamic volatility
        volatility = self._calculate_volatility(recent_prices)

        # Extract the base asset from the symbol (e.g., BTC from BTCUSDT)
        base_asset = symbol.replace("USDT", "")

        try:
            # Get optimized parameters from trade analyzer
            optimized_params = self.trade_analyzer.analyze_trade({
                'symbol': symbol,
                'market_data': {
                    'closes': closing_prices[-20:],
                    'volumes': volumes[-20:],
                    'highs': high_prices[-20:],
                    'lows': low_prices[-20:],
                }
            })

            if optimized_params:
                # Combine base prompt with dynamic learned patterns
                enhanced_prompt = self.base_prompt + "\n" + optimized_params['prompt_template']
            else:
                enhanced_prompt = self.base_prompt

            # Relaxed criteria for more trades
            prompt = f"""
            ## BASIC PRICE ACTION ANALYSIS
            
            You are a beginner crypto trader analyzing {symbol} on a 1-minute timeframe. Use basic price action concepts to identify trading opportunities.
            
            ## PRIMARY GOAL
            Take more trades to gather data for machine learning. Focus on basic setups like support/resistance, trendlines, and simple candlestick patterns.
            
            ## MARKET DATA FOR {symbol}
            • Current Price: {recent_prices[-1]}
            • Recent Closes: {recent_prices[-15:]}
            • Recent Highs: {recent_highs[-10:]}
            • Recent Lows: {recent_lows[-10:]}
            • Current Trend: {trend}
            • Trend Strength: {trend_strength}
            • Volatility: {volatility:.4f}
            
            ## BASIC CRITERIA
            1. Trade with the trend ({trend}).
            2. Look for simple candlestick patterns (e.g., engulfing, pin bars).
            3. Use support/resistance levels for entries and exits.
            4. Allow trades even in moderately choppy markets.
            5. Risk-reward ratio ≥ 1:1.
            
            ## JSON RESPONSE FORMAT
            {{
                "decision": "buy/sell/waiting",
                "reason": "Basic price action analysis",
                "confidence": number, // 1-10 scale
                "entry_price": number,
                "stop_loss": number,
                "take_profit": number
            }}
            """

            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove JSON code block markers if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                decision_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try adding explicit JSON instructions
                clarification_prompt = (
                    "IMPORTANT: Respond with ONLY the JSON object. "
                    "NO markdown formatting, NO code blocks, NO backticks. "
                    "The response must start with '{' and end with '}'."
                )
                response = model.generate_content(prompt + "\n\n" + clarification_prompt)
                response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                
                try:
                    decision_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}\nResponse text: {response_text}")
                    return self._get_default_decision(symbol, closing_prices[-1])

            # Add trend and trend strength to the decision data
            decision_data["trend"] = trend
            decision_data["trend_strength"] = trend_strength

            # Validate the decision data
            if not self._validate_decision(decision_data):
                logger.warning(f"Invalid decision data received for {symbol}: {decision_data}")
                raise ValueError("Invalid decision data structure from Gemini")

            # Log Gemini's decision for analysis
            logger.info(f"Gemini Decision: {decision_data}")
            return decision_data

        except Exception as e:
            logger.error(f"Error in GeminiClient for {symbol}: {e}")
            return self._get_default_decision(symbol, closing_prices[-1])

    def _get_default_decision(self, symbol, current_price):
        """Return a default waiting decision."""
        return {
            "decision": "waiting",
            "reason": "Error processing market data. Defaulting to waiting.",
            "confidence": 0,
            "entry_price": current_price,
            "stop_loss": current_price * 0.99,
            "take_profit": current_price * 1.01,
            "trend": "undefined",
            "trend_strength": "weak",
            "risk_reward_ratio": 1,
            "structure_analysis": {
                "current_structure": "ranging",
                "last_bos_level": None,
                "key_order_blocks": [],
                "liquidity_pools": [],
                "market_structure": {
                    "recent_highs": [],
                    "recent_lows": [],
                    "structure_type": "Ranging",
                    "last_choch": None
                }
            }
        }

    def _validate_decision(self, decision_data: dict) -> bool:
        """Validate the updated decision data returned by Gemini."""
        try:
            required_keys = [
                "decision", "reason", "entry_price", "stop_loss", "take_profit", 
                "trend", "trend_strength", "structure_analysis"
            ]
            
            # Add default values for missing keys
            for key in required_keys:
                if key not in decision_data:
                    if key == "structure_analysis":
                        decision_data[key] = {
                            "current_structure": "undefined",
                            "last_bos_level": None,
                            "key_order_blocks": [],
                            "liquidity_pools": [],
                            "market_structure": {}
                        }
                    else:
                        decision_data[key] = None

            return True
        except Exception as e:
            logger.error(f"Error validating decision data: {e}")
            return False

    def _calculate_trend_strength(self, prices: list[float]) -> str:
        """
        Calculate the strength of the trend based on recent price action.
        Adjusted for 5-minute timeframe to be more sensitive to short-term moves.

        Args:
            prices (list[float]): List of recent closing prices.

        Returns:
            str: Trend strength (weak, moderate, or strong).
        """
        price_change = prices[-1] - prices[0]
        absolute_change = abs(price_change)
        avg_price = sum(prices) / len(prices)
        change_percentage = (absolute_change / avg_price) * 100

        # Adjusted thresholds for 5-minute timeframe
        if change_percentage < 0.15:  # More sensitive to small moves
            return "weak"
        elif 0.15 <= change_percentage < 0.5:  # Moderate moves in 5-minute
            return "moderate"
        else:
            return "strong"

    def _calculate_volatility(self, prices: list[float]) -> float:
        """
        Calculate dynamic volatility based on recent price movements.
        Optimized for 5-minute timeframe.

        Args:
            prices (list[float]): List of recent closing prices.

        Returns:
            float: Volatility factor.
        """
        try:
            # Ensure we have enough data points
            if len(prices) < 22:  # Need at least 22 points for 20-period calculation
                return np.std(np.diff(prices) / prices[:-1])  # Fallback to simple volatility
            
            # Calculate returns for different timeframes
            prices_array = np.array(prices)
            short_term_prices = prices_array[-21:]  # Last 21 prices for 20 periods
            short_term_returns = np.diff(short_term_prices) / short_term_prices[:-1]
            long_term_returns = np.diff(prices_array) / prices_array[:-1]

            # Combine short-term and long-term volatility
            short_term_vol = np.std(short_term_returns) * np.sqrt(20)  # Annualize
            long_term_vol = np.std(long_term_returns) * np.sqrt(len(prices) - 1)

            # Weight recent volatility more heavily
            volatility = (0.7 * short_term_vol + 0.3 * long_term_vol)
            return float(volatility)  # Ensure we return a float
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.001  # Return a default small volatility value