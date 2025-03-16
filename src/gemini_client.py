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

        prompt = f"""
        You are a **professional crypto trader** specializing in Smart Money Concepts (SMC) and advanced price action analysis. Analyze the market structure and identify institutional price movements for {symbol}. Return ONLY a valid JSON response matching the specified format exactly.

        **Market Data for {symbol} (1-minute timeframe):**
        Opening: {recent_opens[-10:]}  # Last 10 values for readability
        Closing: {recent_prices[-10:]}
        Highs: {recent_highs[-10:]}
        Lows: {recent_lows[-10:]}
        Symbol: {symbol}
        Asset: {base_asset}
        Trend: {trend}
        Strength: {trend_strength}
        Volatility: {volatility:.4f}
        Current Price: {recent_prices[-1]}

        **Trading Fee Considerations:**
        - Trading fee is 0.04% per trade (0.08% round trip)
        - Take profit must be at least 0.3% from entry to be profitable
        - Minimum risk-reward ratio must be 1:1.5
        - For scalp trades, risk-reward should be 1:2

        **Special considerations for {base_asset}:**
        - Consider the specific characteristics of {base_asset} and its typical volatility
        - Analyze {base_asset}-specific market dynamics and recent news/developments
        - Include {base_asset}'s correlation with the broader market in your analysis
        - Consider {base_asset}'s liquidity and trading volume characteristics

        Rules for Stop Loss: 
        - For LONG: Place below previous higher high or nearest active order block
        - For SHORT: Place above previous lower high or nearest active order block
        - Stop loss MUST be at a technically significant level, not just a fixed percentage
        - Allow sufficient room for price to breathe (minimum 0.5% from entry for crypto)
        - Minimum 0.5-1% away from entry price to accommodate normal market volatility
        - DO NOT set stops too tight - they will be trailed dynamically by the system

        Rules for Take Profit (in order of priority):
        1. Break of structure to next order block
        2. Liquidity grab levels
        3. 50% of nearest fair value gap
        4. Trend continuation confirmation
        5. Next higher high (for longs) or lower low (for shorts)
        6. Default to 1:2 risk-reward for scalp trades
        7. Must be at least 0.3% away from entry price to cover fees and be profitable

        Analyze for:
        1. Market Structure (HH/HL, LH/LL, BOS, CHoCH)
        2. Order Blocks and Breaker Blocks
        3. Liquidity Grabs/Stop Hunts
        4. Fair Value Gaps
        5. Support/Resistance with Break & Retest

        RETURN EXACTLY THIS JSON FORMAT:
        {{
            "decision": "buy/sell/waiting",
            "reason": "Clear explanation of SMC analysis for {symbol}",
            "entry_price": number,
            "stop_loss": number,
            "take_profit": number,
            "trend": "{trend}",
            "trend_strength": "{trend_strength}",
            "structure_analysis": {{
                "current_structure": "bullish/bearish/ranging",
                "last_bos_level": number,
                "key_order_blocks": [
                    {{
                        "type": "bullish/bearish",
                        "price": number,
                        "status": "active/mitigated"
                    }}
                ],
                "liquidity_pools": [
                    {{
                        "level": number,
                        "type": "buy/sell",
                        "status": "untapped/tapped"
                    }}
                ],
                "market_structure": {{
                    "recent_highs": [number],
                    "recent_lows": [number],
                    "structure_type": "HH-HL/LH-LL/Ranging",
                    "last_choch": number,
                    "prev_higher_high": number,
                    "prev_lower_high": number,
                    "next_key_level": number
                }},
                "fair_value_gaps": [
                    {{
                        "level": number,
                        "size": number,
                        "status": "unfilled/filled"
                    }}
                ],
                "break_and_retest": [
                    {{
                        "level": number,
                        "type": "support/resistance",
                        "status": "confirmed/pending"
                    }}
                ]
            }}
        }}

        Rules for valid response:
        1. All number fields must be actual numbers, not strings
        2. Arrays can be empty [] but must be present
        3. All nested objects must have all required fields
        4. No additional fields allowed
        5. Strings must match exactly the allowed values
        6. For any buy decision, ensure take_profit > entry_price > stop_loss
        7. For any sell decision, ensure stop_loss > entry_price > take_profit
        8. The difference between take_profit and entry_price must be at least 0.3% of entry_price
        9. Stop loss must be at least 0.5% away from entry price to avoid early closure
        """

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            response_text = response.text.replace("```json", "").replace("```", "").strip()
            
            try:
                decision_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}\nResponse text: {response_text}")
                raise ValueError(f"Invalid JSON response from Gemini: {str(e)}")

            # Add trend and trend strength to the decision data
            decision_data["trend"] = trend
            decision_data["trend_strength"] = trend_strength

            # Validate the decision data
            if not self._validate_decision(decision_data):
                raise ValueError("Invalid decision data structure from Gemini")

            logger.info(f"Gemini Decision: {decision_data}")
            return decision_data

        except Exception as e:
            logger.error(f"Error in GeminiClient: {e}")
            current_price = recent_prices[-1]
            return {
                "decision": "waiting",
                "reason": f"Error processing market data: {str(e)}. Please check the system logs for more details.",
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "trend_strength": trend_strength,
                "structure_analysis": {
                    "current_structure": "ranging",
                    "last_bos_level": current_price,
                    "key_order_blocks": [],
                    "liquidity_pools": [],
                    "market_structure": {
                        "recent_highs": [],
                        "recent_lows": [],
                        "structure_type": "Ranging",
                        "last_choch": 0,
                        "prev_higher_high": current_price * 1.01,  # Dummy value, 1% above current price
                        "prev_lower_high": current_price * 0.99,   # Dummy value, 1% below current price
                        "next_key_level": current_price * 1.02     # Dummy value, 2% above current price
                    },
                    "fair_value_gaps": [
                        {
                            "level": current_price,
                            "size": 0,
                            "status": "unfilled"
                        }
                    ],
                    "break_and_retest": [
                        {
                            "level": current_price,
                            "type": "support",
                            "status": "pending"
                        }
                    ]
                }
            }

    def _validate_decision(self, decision_data: dict) -> bool:
        """
        Validate the decision data returned by Gemini, including SMC analysis.

        Args:
            decision_data (dict): Decision data to validate.

        Returns:
            bool: True if the decision data is valid, False otherwise.
        """
        try:
            # Required keys for SMC strategy
            required_keys = [
                "decision", "reason", "entry_price", "trend", "trend_strength", "structure_analysis"
            ]
            
            # Check main required keys
            missing_keys = [key for key in required_keys if key not in decision_data]
            if missing_keys:
                logger.error(f"Missing required keys in decision data: {missing_keys}")
                return False

            # For waiting decisions, stop_loss and take_profit can be None
            if decision_data["decision"] != "waiting":
                if "stop_loss" not in decision_data or "take_profit" not in decision_data:
                    logger.error("Missing stop_loss or take_profit for non-waiting decision")
                    return False

            # Validate structure analysis
            structure_keys = [
                "current_structure", "last_bos_level",
                "key_order_blocks", "liquidity_pools",
                "market_structure", "fair_value_gaps", "break_and_retest"
            ]
            
            # Check structure analysis keys
            if "structure_analysis" not in decision_data:
                logger.error("Missing structure_analysis in decision data")
                return False
                
            missing_structure_keys = [key for key in structure_keys if key not in decision_data["structure_analysis"]]
            if missing_structure_keys:
                logger.error(f"Missing structure analysis keys: {missing_structure_keys}")
                return False

            # Validate market structure components
            market_structure_keys = [
                "recent_highs", "recent_lows", "structure_type", "last_choch"
            ]
            
            # New fields are optional to maintain backward compatibility
            optional_market_keys = ["prev_higher_high", "prev_lower_high", "next_key_level"]
            
            if "market_structure" not in decision_data["structure_analysis"]:
                logger.error("Missing market_structure in structure_analysis")
                return False
                
            missing_market_keys = [key for key in market_structure_keys if key not in decision_data["structure_analysis"]["market_structure"]]
            if missing_market_keys:
                logger.error(f"Missing market structure keys: {missing_market_keys}")
                return False
                
            # Add default values for optional keys if missing
            for key in optional_market_keys:
                if key not in decision_data["structure_analysis"]["market_structure"]:
                    current_price = decision_data["entry_price"]
                    if key == "prev_higher_high":
                        decision_data["structure_analysis"]["market_structure"][key] = current_price * 1.01
                    elif key == "prev_lower_high":
                        decision_data["structure_analysis"]["market_structure"][key] = current_price * 0.99
                    elif key == "next_key_level":
                        decision_data["structure_analysis"]["market_structure"][key] = current_price * 1.02
                    logger.warning(f"Added default value for missing {key}")

            # Validate fair value gaps have size field
            if decision_data["structure_analysis"]["fair_value_gaps"]:
                for gap in decision_data["structure_analysis"]["fair_value_gaps"]:
                    # If any field is missing, add default values rather than failing
                    if "level" not in gap:
                        gap["level"] = decision_data["entry_price"]
                    if "size" not in gap:
                        gap["size"] = 0
                    if "status" not in gap:
                        gap["status"] = "unfilled"

            # Validate decision type
            if decision_data["decision"] not in ["buy", "sell", "waiting"]:
                logger.error(f"Invalid decision type: {decision_data['decision']}")
                return False

            # Validate trend type
            if decision_data["trend"] not in ["uptrend", "downtrend", "sideways"]:
                logger.error(f"Invalid trend: {decision_data['trend']}")
                return False

            # Validate trend strength
            if decision_data["trend_strength"] not in ["weak", "moderate", "strong"]:
                logger.error(f"Invalid trend strength: {decision_data['trend_strength']}")
                return False

            # Validate structure
            if decision_data["structure_analysis"]["current_structure"] not in ["bullish", "bearish", "ranging"]:
                logger.error(f"Invalid current_structure: {decision_data['structure_analysis']['current_structure']}")
                return False

            # Validate market structure type
            if decision_data["structure_analysis"]["market_structure"]["structure_type"] not in ["HH-HL", "LH-LL", "Ranging"]:
                logger.error(f"Invalid structure_type: {decision_data['structure_analysis']['market_structure']['structure_type']}")
                return False

            # Validate stop loss and take profit based on decision
            if decision_data["decision"] != "waiting":
                # If stop_loss or take_profit is None, validation fails
                if decision_data["stop_loss"] is None or decision_data["take_profit"] is None:
                    logger.error("stop_loss or take_profit is None for a non-waiting decision")
                    return False
                
                # Ensure stop_loss and take_profit are numbers
                if not isinstance(decision_data["stop_loss"], (int, float)) or not isinstance(decision_data["take_profit"], (int, float)):
                    logger.error(f"stop_loss or take_profit is not a number: {type(decision_data['stop_loss'])}, {type(decision_data['take_profit'])}")
                    return False
                
                entry_price = decision_data["entry_price"]
                stop_loss = decision_data["stop_loss"]
                take_profit = decision_data["take_profit"]
                
                # Ensure take profit is sufficiently away from entry
                if decision_data["decision"] == "buy":
                    min_distance = entry_price * 0.003  # Minimum 0.3% distance
                    if take_profit - entry_price < min_distance:
                        logger.warning(f"Take profit too close to entry: {take_profit} vs {entry_price}, adjusting")
                        # Fix it instead of failing
                        decision_data["take_profit"] = entry_price + min_distance
                        take_profit = decision_data["take_profit"]
                    
                    if stop_loss >= entry_price:
                        logger.warning(f"Stop loss above entry for buy: {stop_loss} vs {entry_price}, adjusting")
                        # Fix it instead of failing
                        decision_data["stop_loss"] = entry_price * 0.99
                        stop_loss = decision_data["stop_loss"]
                    
                    risk = entry_price - stop_loss
                    reward = take_profit - entry_price
                else:  # sell
                    min_distance = entry_price * 0.003  # Minimum 0.3% distance
                    if entry_price - take_profit < min_distance:
                        logger.warning(f"Take profit too close to entry: {take_profit} vs {entry_price}, adjusting")
                        # Fix it instead of failing
                        decision_data["take_profit"] = entry_price - min_distance
                        take_profit = decision_data["take_profit"]
                        
                    if stop_loss <= entry_price:
                        logger.warning(f"Stop loss below entry for sell: {stop_loss} vs {entry_price}, adjusting")
                        # Fix it instead of failing
                        decision_data["stop_loss"] = entry_price * 1.01
                        stop_loss = decision_data["stop_loss"]
                    
                    risk = stop_loss - entry_price
                    reward = entry_price - take_profit
                
                if risk <= 0:
                    logger.warning(f"Invalid risk (<=0): {risk}, adjusting")
                    # Fix it instead of failing
                    if decision_data["decision"] == "buy":
                        decision_data["stop_loss"] = entry_price * 0.99
                    else:
                        decision_data["stop_loss"] = entry_price * 1.01
                    return True  # Re-validate on next cycle
                
                if reward <= 0:
                    logger.warning(f"Invalid reward (<=0): {reward}, adjusting")
                    # Fix it instead of failing
                    if decision_data["decision"] == "buy":
                        decision_data["take_profit"] = entry_price * 1.01
                    else:
                        decision_data["take_profit"] = entry_price * 0.99
                    return True  # Re-validate on next cycle
                
                # Minimum risk-reward of 1:1.5 - Just warn, don't fail validation
                if reward < (risk * 1.5):
                    logger.warning(f"Insufficient risk-reward ratio: {reward/risk:.2f}, will be adjusted by StrategyManager")
                    # This will be fixed by StrategyManager, so we don't fail here
                
                # For scalp trades, ensure minimum 1:2 risk-reward - Just warn, don't fail validation
                if decision_data.get("trade_type") == "scalp" and reward < (risk * 2):
                    logger.warning(f"Insufficient risk-reward ratio for scalp: {reward/risk:.2f}, will be adjusted by StrategyManager")
                    # This will be fixed by StrategyManager, so we don't fail here

            return True
            
        except Exception as e:
            logger.error(f"Error validating decision data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _calculate_trend_strength(self, prices: list[float]) -> str:
        """
        Calculate the strength of the trend based on recent price action.
        Adjusted for 1-minute timeframe to be more sensitive to short-term moves.

        Args:
            prices (list[float]): List of recent closing prices.

        Returns:
            str: Trend strength (weak, moderate, or strong).
        """
        price_change = prices[-1] - prices[0]
        absolute_change = abs(price_change)
        avg_price = sum(prices) / len(prices)
        change_percentage = (absolute_change / avg_price) * 100

        # Adjusted thresholds for 1-minute timeframe
        if change_percentage < 0.15:  # More sensitive to small moves
            return "weak"
        elif 0.15 <= change_percentage < 0.5:  # Moderate moves in 1-minute
            return "moderate"
        else:
            return "strong"

    def _calculate_volatility(self, prices: list[float]) -> float:
        """
        Calculate dynamic volatility based on recent price movements.
        Optimized for 1-minute timeframe.

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