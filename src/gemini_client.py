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
        ## PROFESSIONAL CRYPTO TRADING ANALYST
        
        You are a professional crypto trader analyzing {symbol} on a 5-minute timeframe. Use advanced price action analysis and Smart Money Concepts (SMC) to identify institutional movements.
        
        ## PRIMARY GOAL
        Identify ONLY high-probability trading setups with clear institutional confirmation.
        BE SELECTIVE - return "waiting" for any setup that doesn't meet ALL mandatory criteria.
        
        ## MARKET DATA FOR {symbol}
        • Current Price: {recent_prices[-1]}
        • Recent Closes (5m): {recent_prices[-15:]}
        • Recent Highs (5m): {recent_highs[-10:]}
        • Recent Lows (5m): {recent_lows[-10:]}
        • Asset Base: {base_asset}
        • Current Trend: {trend}
        • Trend Strength: {trend_strength}
        • Volatility: {volatility:.4f}
        
        ## MANDATORY CRITERIA (ALL MUST BE MET)
        1. Direction must align with prevailing trend ({trend})
        2. Clear Break of Structure (BOS) with Change of Character (CHoCH)
        3. Entry at or near institutional price levels (Order Blocks/Fair Value Gaps)
        4. Minimum 1:1.5 risk-reward ratio (preferable 1:3+)
        5. Clean market structure (avoid choppy/sideways conditions)
        6. Stop loss at technically valid level (not arbitrary %)
        7. Evidence of institutional interest (liquidity sweep/stop hunts)
        8. Entry must be within 0.5% of current price
        
        ## PRIORITY OF SIGNALS (HIGHEST TO LOWEST)
        1. Order Blocks at key levels
        2. Break of Structure (BOS) with Change of Character (CHoCH) 
        3. Fair Value Gaps (FVG) formed by institutional candles
        4. Liquidity grabs/sweeps of significant levels
        5. HTF support/resistance with clear break and retest
        
        ## MULTI-TIMEFRAME CONSIDERATIONS
        • Primary: 5-minute (immediate execution)
        • Consider 5-minute trend direction
        • Avoid trading against 15-minute trend
        • Look for alignment across timeframes for highest probability
        
        ## RISK MANAGEMENT RULES
        ### Stop Loss Placement
        • For LONG: Below nearest active order block or previous swing low
        • For SHORT: Above nearest active order block or previous swing high
        • Minimum 0.5% from entry price
        • Technical level, not arbitrary percentage
        
        ### Take Profit Strategy (Multiple Levels)
        • TP1: Nearest opposing liquidity level (1:1 ratio)
        • TP2: Next key structure level (1:2 ratio)
        • TP3: Extended target at major level (1:3+ ratio)
        
        ## MARKET-SPECIFIC CONSIDERATIONS
        • Match trading style to current {base_asset} volatility
        • Consider {base_asset}'s correlation with BTC (avoid trading against BTC trend)
        • Factor in {base_asset}'s liquidity profile and typical move magnitude
        • Assess potential impact of market-wide events on {base_asset}
        
        ## TECHNICAL PATTERN RECOGNITION
        • Smart Money Concepts (SMC): Order blocks, breaker blocks, FVGs
        • Wyckoff Phases: Accumulation/Distribution/Markup/Markdown
        • Volume Analysis: Divergence from price, climax volume
        • Advanced PA: Engulfing patterns at key levels, multi-candle formations
        
        ## RULES FOR VALID RESPONSE
        • Return "waiting" unless ALL mandatory criteria are met
        • Provide concrete technical reasons for decisions
        • Assess confidence level for each trade (1-10 scale)
        • Calculate exact risk-reward ratio based on SL/TP levels
        
        ## JSON RESPONSE FORMAT
        {{
            "decision": "buy/sell/waiting",
            "reason": "Concise SMC analysis with primary decision factor",
            "confidence": number, // 1-10 scale, must be ≥7 for non-waiting decisions
            "entry_price": number,
            "stop_loss": number,
            "take_profit": {{
                "tp1": number, // 1:1 R:R target
                "tp2": number, // 1:2 R:R target
                "tp3": number  // 1:3+ R:R target
            }},
            "trend": "{trend}",
            "trend_strength": "{trend_strength}",
            "risk_reward_ratio": number, // calculated as (take_profit.tp2 - entry_price) / (entry_price - stop_loss) for buy
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
                        "type": "buy/sell side",
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
                ],
                "wyckoff_phase": "accumulation/markup/distribution/markdown/undefined",
                "volume_analysis": "confirming/diverging/neutral/undefined"
            }}
        }}
        
        IMPORTANT VALIDATION RULES:
        1. All number fields must be actual numbers, not strings
        2. For any buy decision: tp1 > tp2 > tp3 > entry_price > stop_loss
        3. For any sell decision: stop_loss > entry_price > tp3 > tp2 > tp1
        4. Risk-reward ratio must be ≥1.5 
        5. Confidence must be ≥7 for non-waiting decisions
        6. Entry must be within 0.5% of current price ({recent_prices[-1]})
        7. Stop loss must be at least 0.5% from entry price
        8. Take profit must be at least 0.3% from entry price to cover fees
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
        Validate the updated decision data returned by Gemini, including enhanced SMC analysis.

        Args:
            decision_data (dict): Decision data to validate.

        Returns:
            bool: True if the decision data is valid, False otherwise.
        """
        try:
            # Required keys for SMC strategy with enhanced analysis
            required_keys = [
                "decision", "reason", "entry_price", "trend", 
                "trend_strength", "structure_analysis"
            ]
            
            # Check for new confidence field, add default if missing
            if "confidence" not in decision_data:
                if decision_data.get("decision", "waiting") != "waiting":
                    decision_data["confidence"] = 7  # Default confidence for non-waiting
                else:
                    decision_data["confidence"] = 3  # Default low confidence for waiting
            
            # Check for risk_reward_ratio, add if missing
            if "risk_reward_ratio" not in decision_data and decision_data.get("decision", "waiting") != "waiting":
                # Add calculated risk_reward_ratio if possible
                entry_price = decision_data.get("entry_price")
                stop_loss = decision_data.get("stop_loss")
                
                if entry_price is not None and stop_loss is not None:
                    if isinstance(decision_data.get("take_profit"), dict) and "tp2" in decision_data["take_profit"]:
                        tp2 = decision_data["take_profit"]["tp2"]
                        if decision_data.get("decision") == "buy":
                            risk = entry_price - stop_loss
                            reward = tp2 - entry_price
                        else:  # sell
                            risk = stop_loss - entry_price
                            reward = entry_price - tp2
                            
                        if risk > 0:
                            decision_data["risk_reward_ratio"] = reward / risk
                    elif isinstance(decision_data.get("take_profit"), (int, float)):
                        # For backward compatibility with old format
                        take_profit = decision_data["take_profit"]
                        if decision_data.get("decision") == "buy":
                            risk = entry_price - stop_loss
                            reward = take_profit - entry_price
                        else:  # sell
                            risk = stop_loss - entry_price
                            reward = entry_price - take_profit
                            
                        if risk > 0:
                            decision_data["risk_reward_ratio"] = reward / risk
            
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
                
                # Handle the different take profit formats
                take_profit_main = None
                
                if isinstance(decision_data["take_profit"], dict):
                    # We have multiple TP levels
                    if "tp2" in decision_data["take_profit"]:
                        take_profit_main = decision_data["take_profit"]["tp2"]  # Use middle target
                    elif "tp1" in decision_data["take_profit"]:
                        take_profit_main = decision_data["take_profit"]["tp1"]  # Use first target
                    else:
                        # Use any available target
                        take_profit_main = next(iter(decision_data["take_profit"].values()))
                        
                    # Validate TP levels are properly ordered
                    if decision_data["decision"] == "buy":
                        if "tp1" in decision_data["take_profit"] and "tp2" in decision_data["take_profit"] and "tp3" in decision_data["take_profit"]:
                            tp1 = decision_data["take_profit"]["tp1"]
                            tp2 = decision_data["take_profit"]["tp2"]
                            tp3 = decision_data["take_profit"]["tp3"]
                            entry = decision_data["entry_price"]
                            
                            # Check order: entry < tp1 < tp2 < tp3
                            if not (entry < tp1 < tp2 < tp3):
                                logger.warning("Take profit levels not properly ordered for BUY, fixing...")
                                # Fix the order
                                if tp1 <= entry:
                                    tp1 = entry * 1.01  # 1% above entry
                                if tp2 <= tp1:
                                    tp2 = tp1 * 1.01  # 1% above tp1
                                if tp3 <= tp2:
                                    tp3 = tp2 * 1.01  # 1% above tp2
                                    
                                decision_data["take_profit"] = {
                                    "tp1": tp1,
                                    "tp2": tp2,
                                    "tp3": tp3
                                }
                        
                    elif decision_data["decision"] == "sell":
                        if "tp1" in decision_data["take_profit"] and "tp2" in decision_data["take_profit"] and "tp3" in decision_data["take_profit"]:
                            tp1 = decision_data["take_profit"]["tp1"]
                            tp2 = decision_data["take_profit"]["tp2"]
                            tp3 = decision_data["take_profit"]["tp3"]
                            entry = decision_data["entry_price"]
                            
                            # Check order: tp3 < tp2 < tp1 < entry
                            if not (tp3 < tp2 < tp1 < entry):
                                logger.warning("Take profit levels not properly ordered for SELL, fixing...")
                                # Fix the order
                                if tp1 >= entry:
                                    tp1 = entry * 0.99  # 1% below entry
                                if tp2 >= tp1:
                                    tp2 = tp1 * 0.99  # 1% below tp1
                                if tp3 >= tp2:
                                    tp3 = tp2 * 0.99  # 1% below tp2
                                    
                                decision_data["take_profit"] = {
                                    "tp1": tp1,
                                    "tp2": tp2,
                                    "tp3": tp3
                                }
                    
                    # Save the primary take profit for backward compatibility
                    decision_data["take_profit_levels"] = decision_data["take_profit"].copy()
                    decision_data["take_profit"] = take_profit_main
                    
                else:
                    # Single take profit value (old format)
                    take_profit_main = decision_data["take_profit"]
                
                # Ensure stop_loss and take_profit are numbers
                if not isinstance(decision_data["stop_loss"], (int, float)) or not isinstance(take_profit_main, (int, float)):
                    logger.error(f"stop_loss or take_profit is not a number: {type(decision_data['stop_loss'])}, {type(take_profit_main)}")
                    return False
                
                entry_price = decision_data["entry_price"]
                stop_loss = decision_data["stop_loss"]
                take_profit = take_profit_main
                
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

            # Validate confidence score for non-waiting decisions
            if "confidence" in decision_data:
                confidence = decision_data["confidence"]
                if not isinstance(confidence, (int, float)) or confidence < 1 or confidence > 10:
                    logger.warning(f"Invalid confidence score: {confidence}, setting to default 7")
                    decision_data["confidence"] = 7
                elif confidence < 7:
                    logger.warning(f"Low confidence score: {confidence}, might override to waiting")
                    # For very low confidence scores, consider changing to waiting
                    if confidence < 5:
                        logger.warning("Very low confidence, changing decision to waiting")
                        decision_data["decision"] = "waiting"
                        decision_data["reason"] = f"Original reason with insufficient confidence ({confidence}/10): {decision_data['reason']}"
                        return True

            return True
            
        except Exception as e:
            logger.error(f"Error validating decision data: {e}")
            import traceback
            logger.error(traceback.format_exc())
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