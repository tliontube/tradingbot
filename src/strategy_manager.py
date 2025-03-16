import logging
from decimal import Decimal
from datetime import datetime
import math
import time  # Add time import for minimum holding time
import asyncio

class StrategyManager:
    def __init__(self, gemini_client, binance_client, telegram_bot, config=None):
        self.gemini_client = gemini_client
        self.binance_client = binance_client
        self.telegram_bot = telegram_bot
        self.current_trade = None  # Track the current trade
        self.config = config  # Store config as instance variable
        
        # Set default parameter values
        self.position_size = 0.02  # Using 2% of account balance for position sizing
        self.trailing_stop_percent = 0.005  # Increased from 0.001 (0.1%) to 0.005 (0.5%)
        self.trailing_stop_activation = 0.01  # 1% profit before trailing stop activates
        self.min_hold_time_seconds = 300  # 5 minutes minimum hold time
        self.initial_stop_distance = None  # Track initial stop distance
        
        # Fee configuration - defaults
        self.fee_rate = 0.0004  # Default Binance fee rate (0.04%)
        self.min_profit_multiple = 3  # Default minimum profit should be 3x the fees
        self.min_risk_reward = 1.3  # Default minimum risk-reward ratio (changed from 1.5 for more opportunities)
        self.scalp_risk_reward = 2.0  # Default for scalp trades
        
        # Load parameters from config if provided
        if config and 'risk_management' in config:
            risk_config = config['risk_management']
            # Always use 2% of account balance, regardless of config
            self.position_size = 0.02
            self.fee_rate = risk_config.get('fee_rate', 0.0004)
            self.min_profit_multiple = risk_config.get('min_profit_multiple', 3)
            self.min_risk_reward = risk_config.get('min_risk_reward', 1.3)
            self.scalp_risk_reward = risk_config.get('scalp_risk_reward', 2.0)
            # Load new parameters if provided in config
            self.trailing_stop_percent = risk_config.get('trailing_stop_percent', 0.005)
            self.trailing_stop_activation = risk_config.get('trailing_stop_activation', 0.01)
            self.min_hold_time_seconds = risk_config.get('min_hold_time_seconds', 300)
        
        # Also check Binance section for fee rate since it's more specific to the exchange
        if config and 'binance' in config and 'fee_rate' in config['binance']:
            self.fee_rate = config['binance']['fee_rate']
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.initial_balance = 0
        self.current_balance = 0
        self.total_profit_loss = 0
        self.trade_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log the configured parameters
        self.logger.info(f"StrategyManager initialized with: position_size={self.position_size}, " +
                       f"fee_rate={self.fee_rate}, min_profit_multiple={self.min_profit_multiple}, " +
                       f"min_risk_reward={self.min_risk_reward}, trailing_stop_percent={self.trailing_stop_percent}, " +
                       f"trailing_stop_activation={self.trailing_stop_activation}, min_hold_time_seconds={self.min_hold_time_seconds}")

        # Split percentages for the three take-profit levels (must sum to 1.0)
        self.tp_split_percentages = [0.3, 0.3, 0.4]  # 30% at TP1, 30% at TP2, 40% at TP3

    async def initialize_balance_tracking(self):
        """Initialize balance tracking by getting current account balance."""
        balances = await self.binance_client.get_account_balance()
        if balances and "USDT" in balances:
            self.initial_balance = balances["USDT"]["total"]
            self.current_balance = self.initial_balance
            await self.telegram_bot.send_message(
                f"🏦 *Initial Account Balance*: {self.initial_balance:.2f} USDT"
            )

    def calculate_performance_metrics(self):
        """Calculate current performance metrics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        profit_percentage = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_profit_loss": self.total_profit_loss,
            "profit_percentage": profit_percentage
        }

    async def update_performance_metrics(self, trade_result, profit_loss):
        """Update performance metrics after a trade."""
        self.total_trades += 1
        if trade_result == "Win":
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_profit_loss += profit_loss
        self.current_balance += profit_loss
        
        # Add trade to history
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": trade_result,
            "profit_loss": profit_loss,
            "balance": self.current_balance
        })
        
        # Send performance update to Telegram
        await self.send_performance_update()

    async def send_performance_update(self):
        """Send current performance metrics to Telegram."""
        metrics = self.calculate_performance_metrics()
        message = (
            f"📊 *Trading Performance Update* 📊\n\n"
            f"💼 *Total Trades*: {metrics['total_trades']}\n"
            f"✅ *Winning Trades*: {metrics['winning_trades']}\n"
            f"❌ *Losing Trades*: {metrics['losing_trades']}\n"
            f"📈 *Win Rate*: {metrics['win_rate']:.2f}%\n\n"
            f"💰 *Account Summary*:\n"
            f"• Initial Balance: {metrics['initial_balance']:.2f} USDT\n"
            f"• Current Balance: {metrics['current_balance']:.2f} USDT\n"
            f"• Total P/L: {metrics['total_profit_loss']:.2f} USDT\n"
            f"• Return: {metrics['profit_percentage']:.2f}%\n\n"
            f"Keep trading! 📈"
        )
        await self.telegram_bot.send_message(message)

    async def process_trading_decision(self, symbol, gemini_decision):
        """Process the trading decision from Gemini AI."""
        try:
            # If we are already in a trade, don't open a new one
            if self.current_trade is not None:
                return {"decision": "waiting", "reason": "Waiting for current trade to complete", "stop_loss": None, "take_profit": None}

            # Analyze the market and get the decision
            market_data = await self.binance_client.get_market_data(symbol, interval="5m", limit=500)
            if market_data is None or not market_data:
                self.logger.error(f"Unable to fetch market data for {symbol}")
                return {"decision": "waiting", "reason": "Unable to fetch market data", "stop_loss": None, "take_profit": None}
            
            # If we don't have a decision from gemini yet, get one
            if gemini_decision is None:
                gemini_decision = await self.gemini_client.get_trading_decision(
                    market_data["close"], market_data["volume"], 
                    market_data["high"], market_data["low"], 
                    market_data["open"], symbol
                )
                
            if gemini_decision is None:
                return {"decision": "waiting", "reason": "Unable to fetch trading decision", "stop_loss": None, "take_profit": None}

            # If we have a trading decision, validate profit potential considering fees
            if gemini_decision["decision"] in ["buy", "sell"]:
                # Calculate the total fee for a round trip (entry + exit)
                entry_price = gemini_decision["entry_price"]
                stop_loss = gemini_decision["stop_loss"]
                take_profit = gemini_decision["take_profit"]
                
                # Calculate fees for a round trip
                round_trip_fee = entry_price * self.position_size * self.fee_rate * 2  # Entry and exit fees
                
                # Calculate potential profit in base currency
                if gemini_decision["decision"] == "buy":
                    potential_profit = (take_profit - entry_price) * self.position_size
                    potential_loss = (entry_price - stop_loss) * self.position_size
                else:  # sell
                    potential_profit = (entry_price - take_profit) * self.position_size
                    potential_loss = (stop_loss - entry_price) * self.position_size
                
                # Check if profit covers fees with minimum multiple
                if potential_profit <= (round_trip_fee * self.min_profit_multiple):
                    self.logger.info(f"Rejecting trade: Potential profit ({potential_profit:.8f}) too small compared to fees ({round_trip_fee:.8f})")
                    
                    # Adjust take profit to ensure minimum profit
                    if gemini_decision["decision"] == "buy":
                        adjusted_take_profit = entry_price + ((round_trip_fee * self.min_profit_multiple) / self.position_size)
                    else:  # sell
                        adjusted_take_profit = entry_price - ((round_trip_fee * self.min_profit_multiple) / self.position_size)
                    
                    # Update take profit in the decision
                    gemini_decision["take_profit"] = adjusted_take_profit
                    gemini_decision["reason"] += f" (Take profit adjusted to ensure fee coverage)"
                    
                    # Also update take profit levels if present
                    if "take_profit_levels" in gemini_decision and isinstance(gemini_decision["take_profit_levels"], dict):
                        if gemini_decision["decision"] == "buy":
                            # Adjust all TP levels proportionally
                            tp1_adjustment = (adjusted_take_profit - take_profit) * 0.5  # Half the adjustment for TP1
                            tp3_adjustment = (adjusted_take_profit - take_profit) * 1.5  # 1.5x adjustment for TP3
                            
                            gemini_decision["take_profit_levels"]["tp1"] = gemini_decision["take_profit_levels"]["tp1"] + tp1_adjustment
                            gemini_decision["take_profit_levels"]["tp2"] = adjusted_take_profit  # TP2 matches the main take profit
                            gemini_decision["take_profit_levels"]["tp3"] = gemini_decision["take_profit_levels"]["tp3"] + tp3_adjustment
                        else:  # sell
                            tp1_adjustment = (take_profit - adjusted_take_profit) * 0.5  # Half the adjustment for TP1
                            tp3_adjustment = (take_profit - adjusted_take_profit) * 1.5  # 1.5x adjustment for TP3
                            
                            gemini_decision["take_profit_levels"]["tp1"] = gemini_decision["take_profit_levels"]["tp1"] - tp1_adjustment
                            gemini_decision["take_profit_levels"]["tp2"] = adjusted_take_profit  # TP2 matches the main take profit
                            gemini_decision["take_profit_levels"]["tp3"] = gemini_decision["take_profit_levels"]["tp3"] - tp3_adjustment
                    
                    # Check risk-reward ratio after adjustment
                    new_potential_profit = round_trip_fee * self.min_profit_multiple
                    risk_reward_ratio = new_potential_profit / potential_loss
                    
                    if risk_reward_ratio < 1.5:  # Minimum acceptable risk-reward ratio
                        self.logger.info(f"Rejecting trade: Risk-reward ratio too low ({risk_reward_ratio:.2f})")
                        return {"decision": "waiting", "reason": f"Risk-reward ratio too low ({risk_reward_ratio:.2f})", "stop_loss": None, "take_profit": None}
                
                # Log the fees and potential profit
                self.logger.info(f"Trade analysis: Fees={round_trip_fee:.8f}, Potential profit={potential_profit:.8f}, Ratio={potential_profit/round_trip_fee:.2f}x")
                
                # Execute the trade if it passes validation
                trade_executed = await self.execute_trade(symbol, gemini_decision)
                if trade_executed:
                    # Create the current trade record
                    self.current_trade = {
                        "symbol": symbol,
                        "decision": gemini_decision["decision"],
                        "stop_loss": gemini_decision["stop_loss"],
                        "take_profit": gemini_decision["take_profit"],
                        "entry_price": gemini_decision["entry_price"],
                        "position_size": self.position_size,
                        "order_id": trade_executed["orderId"],
                        "entry_fee": entry_price * self.position_size * self.fee_rate,
                        "expected_exit_fee": gemini_decision["take_profit"] * self.position_size * self.fee_rate,
                        "entry_time": time.time()  # Add entry time for minimum hold time check
                    }
                    
                    # Add stop loss order ID if available
                    if "stop_loss_order" in trade_executed and trade_executed["stop_loss_order"]:
                        self.current_trade["stop_loss_order_id"] = trade_executed["stop_loss_order"].get("orderId")
                    
                    # Add take profit order IDs if available - NEW CODE FOR MULTI TP
                    if "take_profit_orders" in trade_executed and trade_executed["take_profit_orders"]:
                        self.current_trade["take_profit_order_ids"] = [
                            order.get("orderId") for order in trade_executed["take_profit_orders"] if order.get("orderId")
                        ]
                        
                        # Also store the take profit levels if they exist
                        if "take_profit_levels" in gemini_decision and isinstance(gemini_decision["take_profit_levels"], dict):
                            self.current_trade["take_profit_levels"] = gemini_decision["take_profit_levels"]
                    elif "take_profit_order" in trade_executed and trade_executed["take_profit_order"]:
                        # For backward compatibility
                        self.current_trade["take_profit_order_id"] = trade_executed["take_profit_order"].get("orderId")
                    
                    # Send Telegram notification about new trade
                    await self.notify_trade_opened(symbol, gemini_decision)
                else:
                    return {"decision": "waiting", "reason": "Trade execution failed", "stop_loss": None, "take_profit": None}

            return gemini_decision
        except Exception as e:
            self.logger.error(f"Error processing trading decision: {e}")
            return {"decision": "waiting", "reason": "Error processing trading decision", "stop_loss": None, "take_profit": None}

    async def execute_trade(self, symbol, decision):
        """Execute a trade based on the strategy decision."""
        try:
            # First check that we're not already in an active trade
            if self.current_trade is not None:
                self.logger.info(f"Already in active trade for {self.current_trade['symbol']}, can't execute new trade")
                return None
                
            # Extract necessary data
            order_side = decision["decision"].upper()  # Convert to uppercase for the API
            price = decision["entry_price"]
            
            # Get symbol information for quantity precision
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Could not get symbol info for {symbol}")
                return None
                
            quantity_precision = symbol_info.get('quantityPrecision', 6) if symbol_info else 6
            
            # Calculate quantity based on position size
            if self.binance_client.using_futures:
                # For futures, we need to convert position size to contract quantity
                # Get current leverage
                leverage = await self.binance_client.get_leverage(symbol)
                if not leverage:
                    self.logger.warning(f"Could not retrieve leverage for {symbol}, using default of 1x")
                    leverage = 1
                    
                self.logger.info(f"Current leverage for {symbol}: {leverage}x")
                    
                # Calculate position value first (position_size is in quote currency for futures)
                latest_price = await self.binance_client.get_latest_price(symbol)
                if latest_price is None or latest_price == 0:
                    latest_price = price  # Fallback to the decision price
                
                # Calculate quantity in base asset
                effective_position = self.position_size * leverage
                quantity = effective_position / latest_price
                
                # Round to the precision required by Binance
                quantity = self._round_to_precision(quantity, quantity_precision)
                
                self.logger.info(f"Calculated futures quantity: {quantity} at leverage {leverage}x (position size: {self.position_size})")
            else:
                # For spot trading, position_size is directly in base currency
                quantity = self.position_size
            
            # Place the order based on the market type
            if self.binance_client.using_futures:
                entry_order = await self.binance_client.place_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="MARKET",
                    quantity=quantity
                )
            else:
                # For spot, use MARKET order
                entry_order = await self.binance_client.place_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="MARKET",
                    quantity=quantity
                )
            
            # Check if we actually got a successful order or error
            if entry_order.get("status") == "ERROR":
                self.logger.error(f"Error placing entry order: {entry_order.get('msg', 'Unknown error')}")
                return None

            self.logger.info(f"Successfully placed {decision['decision']} order: {entry_order}")
            
            # Place stop loss and take profit orders
            stop_loss_order = None
            take_profit_orders = []  # Will store all take profit orders
            
            # Get the execution price from the entry order if available, otherwise use decision["entry_price"]
            entry_price = float(entry_order.get("avgPrice", 0)) 
            if entry_price == 0 or entry_price is None:
                # If avg price not available, try to get the latest price
                entry_price = await self.binance_client.get_latest_price(symbol)
                if entry_price is None or entry_price == 0:
                    entry_price = decision["entry_price"]
            
            # Round prices to the correct precision
            price_precision = symbol_info.get('pricePrecision', 2) if symbol_info else 2
            stop_loss_price = self._round_to_precision(decision["stop_loss"], price_precision)
            
            # Handle multiple take-profit levels if available
            has_multiple_tps = False
            tp_prices = []
            tp_quantities = []
            
            if "take_profit_levels" in decision and isinstance(decision["take_profit_levels"], dict):
                if all(f"tp{i}" in decision["take_profit_levels"] for i in range(1, 4)):
                    has_multiple_tps = True
                    
                    # Get the take profit prices
                    tp1 = self._round_to_precision(decision["take_profit_levels"]["tp1"], price_precision)
                    tp2 = self._round_to_precision(decision["take_profit_levels"]["tp2"], price_precision)
                    tp3 = self._round_to_precision(decision["take_profit_levels"]["tp3"], price_precision)
                    
                    tp_prices = [tp1, tp2, tp3]
                    
                    # Calculate quantities for each TP level based on split percentages
                    total_quantity = quantity
                    for i in range(3):
                        tp_qty = self._round_to_precision(total_quantity * self.tp_split_percentages[i], quantity_precision)
                        # Ensure minimum quantity requirements
                        if tp_qty < float(symbol_info.get('filters', {}).get('LOT_SIZE', {}).get('minQty', 0)):
                            self.logger.warning(f"TP{i+1} quantity {tp_qty} is below minimum. Adjusting split ratios.")
                            # If too small, we'll adjust our strategy
                            has_multiple_tps = False
                            break
                        tp_quantities.append(tp_qty)
            
            # Fallback to single take profit if multiple TPs not available or quantities too small
            if not has_multiple_tps:
                take_profit_price = self._round_to_precision(decision["take_profit"], price_precision)
                tp_prices = [take_profit_price]
                tp_quantities = [quantity]
            
            # For futures trading, need opposite sides for stop loss and take profit
            if self.binance_client.using_futures:
                # For buy (LONG) positions
                if decision["decision"].upper() == "BUY":
                    stop_side = "SELL"
                    tp_side = "SELL"
                # For sell (SHORT) positions
                else:
                    stop_side = "BUY"
                    tp_side = "BUY"
                
                # Get the actual position amount from the exchange 
                position = await self.binance_client.get_position_info(symbol)
                actual_quantity = abs(float(position.get("positionAmt", 0))) if position else quantity
                
                # If position amount is 0, use the original quantity
                if actual_quantity <= 0:
                    self.logger.warning(f"Position amount is 0, using original quantity: {quantity}")
                    actual_quantity = quantity
                
                # Round to precision to avoid quantity errors
                actual_quantity = self._round_to_precision(actual_quantity, quantity_precision)
                
                # If using multiple TPs, adjust quantities proportionally
                if has_multiple_tps:
                    total_quantity = actual_quantity
                    tp_quantities = []
                    for percentage in self.tp_split_percentages:
                        tp_qty = self._round_to_precision(total_quantity * percentage, quantity_precision)
                        tp_quantities.append(tp_qty)
                else:
                    tp_quantities = [actual_quantity]
                
                # Log position details for debugging
                self.logger.info(f"Position info before placing stop loss: {position}")
                
                # Check if we're in hedged mode (BOTH position side)
                correct_position_side = None
                if self.binance_client.futures_position_side == "BOTH":
                    position_amt = float(position.get("positionAmt", 0)) if position else 0
                    # Determine the current position side based on the position amount
                    correct_position_side = "LONG" if position_amt > 0 else "SHORT"
                    self.logger.info(f"Determined position side for stop loss: {correct_position_side} (amount: {position_amt})")
                
                # Try to place stop loss order
                try:
                    # Place stop loss order with retry logic
                    for attempt in range(3):  # Max 3 attempts
                        try:
                            if attempt > 0:
                                self.logger.info(f"Retry #{attempt} for stop loss order")
                                
                            stop_loss_params = {
                                "symbol": symbol,
                                "side": stop_side,
                                "order_type": "STOP_MARKET",
                                "stop_price": stop_loss_price,
                                "quantity": actual_quantity
                            }
                                
                            # Add position side for hedged mode
                            if correct_position_side:
                                stop_loss_params["positionSide"] = correct_position_side
                                
                            stop_loss_order = await self.binance_client.place_order(**stop_loss_params)
                                
                            if stop_loss_order:
                                if stop_loss_order.get("status") == "ERROR":
                                    self.logger.error(f"Error placing stop loss order: {stop_loss_order.get('msg', 'Unknown error')}")
                                    if attempt < 2:  # Try again if not the last attempt
                                        await asyncio.sleep(1)
                                        continue
                                else:
                                    self.logger.info(f"Successfully placed STOP LOSS order at {stop_loss_price}: {stop_loss_order}")
                                    break  # Success, exit retry loop
                            else:
                                self.logger.error(f"Failed to place stop loss order at {stop_loss_price}")
                                if attempt < 2:  # Try again if not the last attempt
                                    await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.error(f"Error on attempt {attempt+1} placing stop loss order: {e}")
                            if "position side does not match" in str(e):
                                # Fix position side mismatch
                                if correct_position_side == "LONG":
                                    correct_position_side = "SHORT"
                                else:
                                    correct_position_side = "LONG"
                                self.logger.info(f"Switching to opposite position side: {correct_position_side}")
                                
                            if attempt < 2:  # Try again if not the last attempt
                                await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error in stop loss order process: {e}")
                
                # Try to place take profit orders
                try:
                    # Wait a moment between orders
                    await asyncio.sleep(1)
                    
                    # Refresh position info
                    position = await self.binance_client.get_position_info(symbol)
                    if not has_multiple_tps:
                        actual_quantity = abs(float(position.get("positionAmt", 0))) if position else quantity
                        
                        # If position amount is 0, use the original quantity
                        if actual_quantity <= 0:
                            self.logger.warning(f"Position amount is 0, using original quantity: {quantity}")
                            actual_quantity = quantity
                        
                        # Round to precision to avoid quantity errors
                        actual_quantity = self._round_to_precision(actual_quantity, quantity_precision)
                        tp_quantities = [actual_quantity]
                    
                    # Get position side for hedged mode
                    correct_position_side = None
                    if self.binance_client.futures_position_side == "BOTH":
                        position_amt = float(position.get("positionAmt", 0)) if position else 0
                        correct_position_side = "LONG" if position_amt > 0 else "SHORT"
                        self.logger.info(f"Determined position side for take profit: {correct_position_side} (amount: {position_amt})")
                    
                    # Place take profit orders
                    for i, (tp_price, tp_qty) in enumerate(zip(tp_prices, tp_quantities)):
                        # Place take profit order with retry logic
                        for attempt in range(3):  # Max 3 attempts
                            try:
                                if attempt > 0:
                                    self.logger.info(f"Retry #{attempt} for TP{i+1} order")
                                    
                                take_profit_params = {
                                    "symbol": symbol,
                                    "side": tp_side,
                                    "order_type": "TAKE_PROFIT_MARKET",
                                    "stop_price": tp_price,
                                    "quantity": tp_qty
                                }
                                
                                # Add position side for hedged mode
                                if correct_position_side:
                                    take_profit_params["positionSide"] = correct_position_side
                                
                                take_profit_order = await self.binance_client.place_order(**take_profit_params)
                                
                                if take_profit_order:
                                    if take_profit_order.get("status") == "ERROR":
                                        self.logger.error(f"Error placing TP{i+1} order: {take_profit_order.get('msg', 'Unknown error')}")
                                        if attempt < 2:  # Try again if not the last attempt
                                            await asyncio.sleep(1)
                                            continue
                                    else:
                                        self.logger.info(f"Successfully placed TP{i+1} order at {tp_price}: {take_profit_order}")
                                        take_profit_orders.append(take_profit_order)
                                        break  # Success, exit retry loop
                                else:
                                    self.logger.error(f"Failed to place TP{i+1} order at {tp_price}")
                                    if attempt < 2:  # Try again if not the last attempt
                                        await asyncio.sleep(1)
                            except Exception as e:
                                self.logger.error(f"Error on attempt {attempt+1} placing TP{i+1} order: {e}")
                                if "position side does not match" in str(e):
                                    # Fix position side mismatch
                                    if correct_position_side == "LONG":
                                        correct_position_side = "SHORT"
                                    else:
                                        correct_position_side = "LONG"
                                    self.logger.info(f"Switching to opposite position side: {correct_position_side}")
                                    
                                if attempt < 2:  # Try again if not the last attempt
                                    await asyncio.sleep(1)
                        
                        # Wait a moment between placing TP orders
                        await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error in take profit order process: {e}")
            
            # Add the orders to the entry_order response for processing
            if stop_loss_order:
                entry_order["stop_loss_order"] = stop_loss_order
            
            if take_profit_orders:
                entry_order["take_profit_orders"] = take_profit_orders
            
            return entry_order
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _round_to_precision(self, value, precision):
        """Round a value to the specified decimal precision."""
        factor = 10 ** precision
        return math.floor(value * factor) / factor

    async def check_trade_completion(self, symbol=None):
        """
        Check if the current trade is complete.
        If symbol is provided, check just that symbol.
        If no symbol is provided, check the current active trade.
        """
        if self.current_trade is None:
            return True

        # Get the symbol from the current trade if not provided
        symbol = symbol or self.current_trade.get("symbol")
        if not symbol:
            self.logger.error("No symbol specified and no current trade symbol available")
            return False

        latest_price = await self.binance_client.get_latest_price(symbol)
        if latest_price is None:
            return False

        stop_loss = self.current_trade["stop_loss"]
        take_profit = self.current_trade["take_profit"]
        entry_price = self.current_trade["entry_price"]
        
        # Check minimum hold time
        current_time = time.time()
        if current_time - self.current_trade["entry_time"] < self.min_hold_time_seconds:
            # Log that we're still in minimum hold time
            self.logger.info(f"Trade still in minimum hold time. Elapsed: {int(current_time - self.current_trade['entry_time'])}s, Required: {self.min_hold_time_seconds}s")
            return False

        # For futures trading, get current position information
        if self.binance_client.using_futures:
            position = await self.binance_client.get_position_info(symbol)
            
            # Check if there are any open orders for this symbol
            open_orders = await self.binance_client.get_open_orders(symbol)
            
            # If we have stop loss and take profit order IDs stored, check if they're still active
            stop_loss_active = False
            take_profit_active = False
            
            if open_orders:
                for order in open_orders:
                    if str(order.get("orderId")) == str(self.current_trade.get("stop_loss_order_id")):
                        stop_loss_active = True
                    if str(order.get("orderId")) == str(self.current_trade.get("take_profit_order_id")):
                        take_profit_active = True
            
            # If position is closed but we still have open orders, cancel them
            if not position or float(position.get("positionAmt", 0)) == 0:
                # No position or position already closed
                self.logger.info(f"No active futures position found for {symbol}")
                
                # Cancel any remaining stop loss or take profit orders
                if stop_loss_active or take_profit_active:
                    self.logger.info(f"Canceling remaining orders for {symbol}")
                    await self.cancel_all_open_orders(symbol)
                
                self.current_trade = None
                return True
                
            # Get unrealized PnL percentage
            entry_price = float(position.get("entryPrice", entry_price))
            position_amount = float(position.get("positionAmt", 0))
            position_side = "buy" if position_amount > 0 else "sell"
            
            # Calculate unrealized PnL percentage
            if position_side == "buy":
                unrealized_pnl_pct = (latest_price - entry_price) / entry_price
            else:
                unrealized_pnl_pct = (entry_price - latest_price) / entry_price
                
            self.logger.info(f"Futures position: Entry={entry_price}, Current={latest_price}, UnrealizedPnL={unrealized_pnl_pct*100:.2f}%")
                
            # Update trailing stop if needed
            if position_side == "buy":
                # Only move stop if price has moved enough in our favor (activation threshold)
                if unrealized_pnl_pct > self.trailing_stop_activation:
                    # Calculate potential new stop loss based on trailing stop
                    potential_stop = latest_price * (1 - self.trailing_stop_percent)
                    if potential_stop > stop_loss:  # Move stop loss up
                        old_stop = stop_loss
                        self.current_trade["stop_loss"] = potential_stop
                        stop_loss = potential_stop
                        await self.notify_stop_loss_update(symbol, stop_loss)
                        self.logger.info(f"Trailing stop updated to: {stop_loss} (original entry: {entry_price})")
                        
                        # If we have an active stop loss order, update it
                        if stop_loss_active:
                            # Cancel the old stop loss order
                            await self.binance_client.cancel_order(symbol, self.current_trade.get("stop_loss_order_id"))
                            
                            # Place a new stop loss order
                            try:
                                stop_loss_order = await self.binance_client.place_order(
                                    symbol=symbol,
                                    side="SELL",
                                    order_type="STOP_MARKET",
                                    stop_price=self._round_to_precision(stop_loss, self.binance_client.get_price_precision(symbol)),
                                    quantity=abs(position_amount),
                                    positionSide="LONG" if self.binance_client.futures_position_side == "BOTH" else None,
                                    reduce_only=True
                                )
                                
                                if stop_loss_order:
                                    self.current_trade["stop_loss_order_id"] = stop_loss_order.get("orderId")
                                    self.logger.info(f"Updated stop loss order: {stop_loss_order}")
                            except Exception as e:
                                self.logger.error(f"Error updating stop loss order: {e}")

                # If the market has triggered our stop loss or take profit, close the position
                if (latest_price <= stop_loss or latest_price >= take_profit) and not (stop_loss_active and take_profit_active):
                    close_order = await self.close_position(symbol, "SELL")
                    if not close_order:
                        return False

                    result = "Loss" if latest_price <= stop_loss else "Win"
                    await self.notify_trade_closed(symbol, result, entry_price, latest_price)
                    self.current_trade = None
                    return True

            elif position_side == "sell":
                # Only move stop if price has moved enough in our favor (activation threshold)
                if unrealized_pnl_pct > self.trailing_stop_activation:
                    # Calculate potential new stop loss based on trailing stop
                    potential_stop = latest_price * (1 + self.trailing_stop_percent)
                    if potential_stop < stop_loss:  # Move stop loss down
                        old_stop = stop_loss
                        self.current_trade["stop_loss"] = potential_stop
                        stop_loss = potential_stop
                        await self.notify_stop_loss_update(symbol, stop_loss)
                        self.logger.info(f"Trailing stop updated to: {stop_loss} (original entry: {entry_price})")
                        
                        # If we have an active stop loss order, update it
                        if stop_loss_active:
                            # Cancel the old stop loss order
                            await self.binance_client.cancel_order(symbol, self.current_trade.get("stop_loss_order_id"))
                            
                            # Place a new stop loss order
                            try:
                                stop_loss_order = await self.binance_client.place_order(
                                    symbol=symbol,
                                    side="BUY",
                                    order_type="STOP_MARKET",
                                    stop_price=self._round_to_precision(stop_loss, self.binance_client.get_price_precision(symbol)),
                                    quantity=abs(position_amount),
                                    positionSide="SHORT" if self.binance_client.futures_position_side == "BOTH" else None,
                                    reduce_only=True
                                )
                                
                                if stop_loss_order:
                                    self.current_trade["stop_loss_order_id"] = stop_loss_order.get("orderId")
                                    self.logger.info(f"Updated stop loss order: {stop_loss_order}")
                            except Exception as e:
                                self.logger.error(f"Error updating stop loss order: {e}")

                # If the market has triggered our stop loss or take profit, close the position
                if (latest_price >= stop_loss or latest_price <= take_profit) and not (stop_loss_active and take_profit_active):
                    close_order = await self.close_position(symbol, "BUY")
                    if not close_order:
                        return False

                    result = "Loss" if latest_price >= stop_loss else "Win"
                    await self.notify_trade_closed(symbol, result, entry_price, latest_price)
                    self.current_trade = None
                    return True
        else:
            # Original spot trading logic
            # Update trailing stop if needed
            if self.current_trade["decision"] == "buy":
                # Only move stop if price has moved enough in our favor (activation threshold)
                if latest_price > entry_price * (1 + self.trailing_stop_activation):
                    # Calculate potential new stop loss based on trailing stop
                    potential_stop = latest_price * (1 - self.trailing_stop_percent)
                    if potential_stop > stop_loss:  # Move stop loss up
                        self.current_trade["stop_loss"] = potential_stop
                        stop_loss = potential_stop
                        await self.notify_stop_loss_update(symbol, stop_loss)
                        self.logger.info(f"Trailing stop updated to: {stop_loss} (original entry: {entry_price})")

                if latest_price <= stop_loss or latest_price >= take_profit:
                    close_order = await self.close_position(symbol, "SELL")
                    if not close_order:
                        return False

                    result = "Loss" if latest_price <= stop_loss else "Win"
                    await self.notify_trade_closed(symbol, result, entry_price, latest_price)
                    self.current_trade = None
                    return True

            elif self.current_trade["decision"] == "sell":
                # Only move stop if price has moved enough in our favor (activation threshold)
                if latest_price < entry_price * (1 - self.trailing_stop_activation):
                    # Calculate potential new stop loss based on trailing stop
                    potential_stop = latest_price * (1 + self.trailing_stop_percent)
                    if potential_stop < stop_loss:  # Move stop loss down
                        self.current_trade["stop_loss"] = potential_stop
                        stop_loss = potential_stop
                        await self.notify_stop_loss_update(symbol, stop_loss)
                        self.logger.info(f"Trailing stop updated to: {stop_loss} (original entry: {entry_price})")

                if latest_price >= stop_loss or latest_price <= take_profit:
                    close_order = await self.close_position(symbol, "BUY")
                    if not close_order:
                        return False

                    result = "Loss" if latest_price >= stop_loss else "Win"
                    await self.notify_trade_closed(symbol, result, entry_price, latest_price)
                    self.current_trade = None
                    return True

        return False

    async def close_position(self, symbol, side):
        """Close an open position."""
        try:
            # First, cancel any existing stop loss and take profit orders
            await self.cancel_all_open_orders(symbol)
            
            # Get symbol info to determine the correct precision for quantity
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Could not fetch symbol info for {symbol}")
                # Use default precision if symbol info not available
                quantity_precision = 5
            else:
                # Get the precision for quantity from symbol info
                quantity_precision = symbol_info.get('quantityPrecision', 5)
            
            if self.binance_client.using_futures:
                # For futures, get the current position size
                position = await self.binance_client.get_position_info(symbol)
                if position:
                    quantity = abs(float(position.get("positionAmt", 0)))
                    if quantity > 0:
                        quantity = self._round_to_precision(quantity, quantity_precision)
                        
                        # Determine position side if in hedged mode
                        position_params = {}
                        if self.binance_client.futures_position_side == "BOTH":
                            pos_amount = float(position.get("positionAmt", 0))
                            position_side = "LONG" if pos_amount > 0 else "SHORT"
                            self.logger.info(f"Closing {position_side} position with {side} order")
                            position_params["positionSide"] = position_side
                        
                        self.logger.info(f"Closing futures position for {symbol} with {side} order: quantity={quantity}")
                        
                        order = await self.binance_client.place_order(
                            symbol=symbol,
                            side=side,
                            order_type="MARKET",
                            quantity=quantity,
                            reduce_only=True,
                            **position_params
                        )
                        return order
                    else:
                        self.logger.warning(f"No position to close for {symbol}")
                        return {"status": "NO_POSITION"}
            else:
                # For spot trading, use the stored position size
                quantity = self.current_trade["position_size"]
                quantity = self._round_to_precision(quantity, quantity_precision)
                
                self.logger.info(f"Closing spot position for {symbol} with {side} order: quantity={quantity}")
                
                order = await self.binance_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="MARKET",
                    quantity=quantity
                )
                return order
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None

    async def cancel_all_open_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            # First, cancel take profit orders if any
            if self.current_trade:
                # Handle multiple take-profit orders
                if "take_profit_order_ids" in self.current_trade and self.current_trade["take_profit_order_ids"]:
                    for tp_order_id in self.current_trade["take_profit_order_ids"]:
                        if tp_order_id:
                            self.logger.info(f"Canceling take profit order: {tp_order_id}")
                            await self.binance_client.cancel_order(symbol, tp_order_id)
                # Backward compatibility for single take-profit
                elif "take_profit_order_id" in self.current_trade and self.current_trade["take_profit_order_id"]:
                    take_profit_order_id = self.current_trade["take_profit_order_id"]
                    self.logger.info(f"Canceling take profit order: {take_profit_order_id}")
                    await self.binance_client.cancel_order(symbol, take_profit_order_id)

                # Cancel stop loss order if exists
                if "stop_loss_order_id" in self.current_trade and self.current_trade["stop_loss_order_id"]:
                    stop_loss_order_id = self.current_trade["stop_loss_order_id"]
                    self.logger.info(f"Canceling stop loss order: {stop_loss_order_id}")
                    await self.binance_client.cancel_order(symbol, stop_loss_order_id)
            
            # Cancel all other open orders
            await self.binance_client.cancel_all_open_orders(symbol)
            self.logger.info(f"All open orders for {symbol} canceled")
            return True
        except Exception as e:
            self.logger.error(f"Error canceling open orders: {e}")
            return False

    async def notify_trade_opened(self, symbol, decision):
        """Send notification when a new trade is opened."""
        try:
            if not self.telegram_bot:
                return
                
            emoji_prefix = "🚀 BUY" if decision["decision"] == "buy" else "🛑 SELL"
            
            # Determine take-profit message format
            take_profit_message = ""
            if "take_profit_levels" in decision and isinstance(decision["take_profit_levels"], dict) and "tp1" in decision["take_profit_levels"]:
                take_profit_message = (
                    f"🎯 **Take-Profit Levels**:\n"
                    f"   • TP1 (1:1): {decision['take_profit_levels']['tp1']} " +
                    (f"(automatic order placed for {self.tp_split_percentages[0]*100}% of position)" 
                     if self.current_trade and "take_profit_order_ids" in self.current_trade else "(manual monitoring)") +
                    f"\n   • TP2 (1:2): {decision['take_profit_levels']['tp2']} " +
                    (f"(automatic order placed for {self.tp_split_percentages[1]*100}% of position)" 
                     if self.current_trade and "take_profit_order_ids" in self.current_trade else "(manual monitoring)") +
                    f"\n   • TP3 (1:3+): {decision['take_profit_levels']['tp3']} " +
                    (f"(automatic order placed for {self.tp_split_percentages[2]*100}% of position)" 
                     if self.current_trade and "take_profit_order_ids" in self.current_trade else "(manual monitoring)") +
                    f"\n"
                )
            else:
                take_profit_message = (
                    f"🎯 **Take-Profit**: {decision['take_profit']} " +
                    (f"(automatic order placed)" if self.current_trade and 
                     ("take_profit_order_id" in self.current_trade or "take_profit_order_ids" in self.current_trade) 
                     else "(manual monitoring)") +
                    f"\n"
                )
            
            # Calculate risk-reward ratio
            risk_reward = "N/A"
            if decision["decision"] == "buy":
                risk = decision["entry_price"] - decision["stop_loss"]
                reward = decision["take_profit"] - decision["entry_price"]
                if risk > 0:
                    risk_reward = f"{(reward / risk):.2f}"
            else:
                risk = decision["stop_loss"] - decision["entry_price"]
                reward = decision["entry_price"] - decision["take_profit"]
                if risk > 0:
                    risk_reward = f"{(reward / risk):.2f}"

            message = (
                f"**{emoji_prefix} SIGNAL for {symbol}**\n\n"
                f"💲 **Entry Price**: {decision['entry_price']}\n"
                f"🛡️ **Stop-Loss**: {decision['stop_loss']} " +
                (f"(automatic order placed)" if self.current_trade and "stop_loss_order_id" in self.current_trade else "(manual monitoring)") +
                f"\n"
                f"{take_profit_message}"
                f"📈 **Risk-Reward Ratio**: {risk_reward}\n\n"
                f"📊 **Analysis**: {decision['reason']}\n"
            )
            
            # Add confidence score if available
            if "confidence" in decision:
                message += f"🧠 **Confidence Score**: {decision['confidence']}/10\n"
            
            # Add structure analysis summary if available
            if "structure_analysis" in decision and isinstance(decision["structure_analysis"], dict):
                structure = decision["structure_analysis"].get("current_structure", "Unknown")
                market_structure = decision["structure_analysis"].get("market_structure", {})
                structure_type = market_structure.get("structure_type", "Unknown")
                
                message += f"\n📐 **Market Structure**: {structure.capitalize()} ({structure_type})\n"
                
                # Add Wyckoff phase if available
                wyckoff = decision["structure_analysis"].get("wyckoff_phase", "undefined")
                if wyckoff != "undefined":
                    message += f"📉 **Wyckoff Phase**: {wyckoff.capitalize()}\n"
                
                # Add volume analysis if available
                volume = decision["structure_analysis"].get("volume_analysis", "undefined")
                if volume != "undefined":
                    message += f"📊 **Volume Analysis**: {volume.capitalize()}\n"
            
            await self.telegram_bot.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade opened notification: {e}")

    async def notify_trade_closed(self, symbol, result, entry_price, exit_price):
        """Send notification when a trade is closed."""
        # Calculate actual profit/loss with fees
        entry_fee = self.current_trade.get("entry_fee", entry_price * self.position_size * self.fee_rate)
        exit_fee = exit_price * self.position_size * self.fee_rate
        total_fees = entry_fee + exit_fee
        
        # Calculate raw profit/loss
        raw_profit_loss = abs(exit_price - entry_price) * self.position_size
        if result == "Loss":
            raw_profit_loss = -raw_profit_loss
        
        # Actual profit/loss after fees
        actual_profit_loss = raw_profit_loss - total_fees
        
        # Update performance metrics with actual profit/loss
        await self.update_performance_metrics(result, actual_profit_loss)
        
        # Determine how the trade was closed
        close_reason = ""
        if result == "Win":
            if exit_price >= self.current_trade.get("take_profit", 0) and self.current_trade.get("decision") == "buy":
                close_reason = "Take profit target reached ✅"
            elif exit_price <= self.current_trade.get("take_profit", 0) and self.current_trade.get("decision") == "sell":
                close_reason = "Take profit target reached ✅"
            else:
                close_reason = "Trailing stop triggered (profit) 📈"
        else:  # Loss
            if exit_price <= self.current_trade.get("stop_loss", 0) and self.current_trade.get("decision") == "buy":
                close_reason = "Stop loss triggered 🛑"
            elif exit_price >= self.current_trade.get("stop_loss", 0) and self.current_trade.get("decision") == "sell":
                close_reason = "Stop loss triggered 🛑"
            else:
                close_reason = "Manual intervention or other reason ⚠️"
        
        # Calculate percentage gain/loss
        percent_change = ((exit_price - entry_price) / entry_price) * 100
        if self.current_trade.get("decision") == "sell":
            percent_change = -percent_change
        
        message = (
            f"📢 *Trade Closed* 📢\n\n"
            f"📊 **Symbol**: {symbol}\n"
            f"📝 **Result**: {result}\n"
            f"🔍 **Close Reason**: {close_reason}\n"
            f"💰 **Entry Price**: {entry_price}\n"
            f"📈 **Exit Price**: {exit_price}\n"
            f"📊 **Change**: {percent_change:.2f}%\n"
            f"💵 **Raw P/L**: {raw_profit_loss:.8f} USDT\n"
            f"💸 **Fees Paid**: {total_fees:.8f} USDT\n"
            f"💹 **Net P/L**: {actual_profit_loss:.8f} USDT\n\n"
            f"{'Better luck next time! 🚀' if result == 'Loss' else 'Great job! 🎉'}"
        )
        await self.telegram_bot.send_message(message)

    async def notify_stop_loss_update(self, symbol, new_stop_loss):
        """Send notification when trailing stop is updated."""
        message = (
            f"🔄 *Trailing Stop Updated* 🔄\n\n"
            f"📊 **Symbol**: {symbol}\n"
            f"🛑 **New Stop-Loss**: {new_stop_loss:.8f}\n\n"
            f"Protecting profits! 🛡️"
        )
        await self.telegram_bot.send_message(message)

    async def find_best_opportunity(self, trading_pairs):
        """
        Analyze all trading pairs and find the best opportunity.
        
        Args:
            trading_pairs (list): List of trading pairs to analyze
            
        Returns:
            dict: Best trading opportunity, or None if no good opportunities found
        """
        opportunities = []
        
        for symbol in trading_pairs:
            try:
                market_data = await self.binance_client.get_market_data(symbol, interval="5m", limit=200)
                if market_data is None:
                    self.logger.warning(f"Could not fetch market data for {symbol}")
                    continue
                    
                decision = self.gemini_client.get_trading_decision(
                    closing_prices=market_data["closing_prices"],
                    volumes=market_data["volumes"],
                    high_prices=market_data["high_prices"],
                    low_prices=market_data["low_prices"],
                    open_prices=market_data["open_prices"],
                    symbol=symbol
                )
                
                if decision is None:
                    self.logger.warning(f"Could not get trading decision for {symbol}")
                    continue
                
                # Calculate opportunity score
                score = self._calculate_opportunity_score(symbol, decision, market_data)
                
                opportunities.append({
                    "symbol": symbol,
                    "decision_data": decision,
                    "score": score
                })
                
                self.logger.info(f"Symbol: {symbol}, Decision: {decision['decision']}, Score: {score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        if not opportunities:
            self.logger.warning("No trading opportunities found")
            return None
            
        # Sort opportunities by score (highest first)
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        # Check if the best opportunity passes the minimum required score threshold
        # Setting a moderate threshold for balanced selectivity
        minimum_threshold = 65.0  # Changed from 85.0 to 65.0 for moderate selectivity
        
        best_opportunity = opportunities[0]
        
        if best_opportunity["score"] < minimum_threshold:
            self.logger.info(f"Best opportunity {best_opportunity['symbol']} with score {best_opportunity['score']:.2f} did not meet minimum threshold of {minimum_threshold}")
            # Convert to a waiting decision if score is below threshold
            best_opportunity["decision_data"]["decision"] = "waiting"
            best_opportunity["decision_data"]["reason"] = f"Waiting for better setup - current score: {best_opportunity['score']:.2f}, minimum required: {minimum_threshold}"
        else:
            self.logger.info(f"Best opportunity: {best_opportunity['symbol']} with score {best_opportunity['score']:.2f}")
            
        return best_opportunity

    def _calculate_opportunity_score(self, symbol, decision, market_data):
        """
        Calculate a score for a trading opportunity based on various factors.
        Enhanced to identify perfect trading setups with professional criteria.
        
        Args:
            symbol (str): The trading pair symbol
            decision (dict): The trading decision
            market_data (dict): Market data for the symbol
            
        Returns:
            float: Opportunity score (higher is better)
        """
        # Initialize base score
        score = 30.0  # Lower base score to make the bot even more selective
        
        # Skip if not a buy or sell decision
        if decision["decision"] not in ["buy", "sell"]:
            return 0.0
            
        # Extract necessary data
        entry_price = decision["entry_price"]
        stop_loss = decision["stop_loss"]
        take_profit = decision["take_profit"]
        trend = decision.get("trend", "unknown")
        trend_strength = decision.get("trend_strength", "unknown")
        
        # Extract structure analysis if available
        structure_analysis = decision.get("structure_analysis", {})
        current_structure = structure_analysis.get("current_structure", "ranging")
        
        # Calculate price movement required for profit (from entry to take profit)
        if decision["decision"] == "buy":
            price_movement = (take_profit - entry_price) / entry_price
        else:  # sell
            price_movement = (entry_price - take_profit) / entry_price
            
        # Calculate potential risk (from entry to stop loss)
        if decision["decision"] == "buy":
            risk = (entry_price - stop_loss) / entry_price
        else:  # sell
            risk = (stop_loss - entry_price) / entry_price
            
        # Calculate risk-reward ratio
        if risk > 0:
            risk_reward = price_movement / risk
        else:
            risk_reward = 0
            
        # Calculate fees for a round trip
        round_trip_fee = entry_price * self.position_size * self.fee_rate * 2
        
        # Calculate potential profit in base currency
        potential_profit = price_movement * self.position_size * entry_price
        
        # ====== ENHANCED PROFESSIONAL SCORING FACTORS ======
        
        # 1. Perfect alignment with trend and market structure (higher score for perfect alignment)
        # This is the foundation of professional trading - trend alignment
        if trend == "uptrend" and decision["decision"] == "buy" and current_structure == "bullish":
            score += 20  # Perfect alignment
        elif trend == "downtrend" and decision["decision"] == "sell" and current_structure == "bearish":
            score += 20  # Perfect alignment
        elif trend == "uptrend" and decision["decision"] == "buy":
            score += 10  # Partial alignment (trend only)
        elif trend == "downtrend" and decision["decision"] == "sell":
            score += 10  # Partial alignment (trend only)
        elif current_structure == "bullish" and decision["decision"] == "buy":
            score += 8  # Partial alignment (structure only)
        elif current_structure == "bearish" and decision["decision"] == "sell":
            score += 8  # Partial alignment (structure only)
        elif trend == "ranging":
            score -= 15  # Significant penalty for ranging market
        else:
            # Counter-trend trades are risky
            score -= 10  # Increased penalty for counter-trend trades
            
        # 2. Trend strength - professional traders wait for strong trends
        if trend_strength == "strong":
            score += 18  # Increased importance of strong trends
        elif trend_strength == "moderate":
            score += 5
        elif trend_strength == "weak":
            score -= 15  # Stronger penalty for weak trends
            
        # 3. Risk-reward ratio - professional traders require excellent R:R
        if risk_reward >= 3.5:
            score += 30  # Exceptional risk-reward
        elif risk_reward >= 3:
            score += 25  # Excellent risk-reward
        elif risk_reward >= 2.5:
            score += 20  # Very good risk-reward
        elif risk_reward >= 2:
            score += 10  # Good risk-reward
        elif risk_reward >= 1.5:
            score += 5  # Minimum acceptable risk-reward
        else:
            score -= 20  # Severe penalty for poor risk/reward
            
        # 4. Profit potential relative to fees - professional traders ensure good profit margin
        profit_multiple = potential_profit / round_trip_fee if round_trip_fee > 0 else 0
        if profit_multiple >= 8:
            score += 25  # Exceptional profit potential
        elif profit_multiple >= 5:
            score += 15  # Excellent profit potential
        elif profit_multiple >= 3:
            score += 8  # Good profit potential
        elif profit_multiple >= 2:
            score += 3  # Minimal profit potential
        else:
            score -= 12  # Stronger penalty for small profit potential
            
        # 5. Perfect market structure alignment with key levels
        market_structure = structure_analysis.get("market_structure", {})
        key_order_blocks = structure_analysis.get("key_order_blocks", [])
        last_bos_level = structure_analysis.get("last_bos_level", None)
        
        # Check if we have a confirmed Break of Structure (BOS)
        if last_bos_level is not None:
            if decision["decision"] == "buy" and last_bos_level < entry_price:
                score += 8  # Bullish BOS confirmed
            elif decision["decision"] == "sell" and last_bos_level > entry_price:
                score += 8  # Bearish BOS confirmed
                
        # Check if we're trading with active order blocks
        for block in key_order_blocks:
            if block.get("status") == "active":
                if block.get("type") == "bullish" and decision["decision"] == "buy":
                    score += 10  # Trading from bullish order block
                elif block.get("type") == "bearish" and decision["decision"] == "sell":
                    score += 10  # Trading from bearish order block
                    
        # Structure pattern recognition
        structure_type = market_structure.get("structure_type", "")
        if structure_type == "HH-HL" and decision["decision"] == "buy":
            score += 15  # Strong bullish structure - higher highs, higher lows
        elif structure_type == "LH-LL" and decision["decision"] == "sell":
            score += 15  # Strong bearish structure - lower highs, lower lows
            
        # 6. Check for Change of Character (CHoCH) - a powerful professional entry signal
        choch_level = market_structure.get("last_choch")
        if choch_level:
            if decision["decision"] == "buy" and abs(entry_price - choch_level) / entry_price < 0.01:
                score += 20  # Trading at CHoCH level (buy)
            elif decision["decision"] == "sell" and abs(entry_price - choch_level) / entry_price < 0.01:
                score += 20  # Trading at CHoCH level (sell)
            
        # 7. Liquidity grab setups - professional traders look for liquidity traps
        liquidity_pools = structure_analysis.get("liquidity_pools", [])
        for pool in liquidity_pools:
            if pool.get("status") == "untapped":
                level = pool.get("level", 0)
                # If we're taking a trade near an untapped liquidity pool
                pool_distance = abs(entry_price - level) / entry_price
                if pool_distance < 0.015:  # Within 1.5% of an untapped liquidity pool
                    if pool.get("type") == "buy" and decision["decision"] == "buy":
                        score += 12  # Buy at a buy liquidity level
                    elif pool.get("type") == "sell" and decision["decision"] == "sell":
                        score += 12  # Sell at a sell liquidity level
            elif pool.get("status") == "tapped":
                # Recently tapped liquidity is also a good sign for continuation
                if pool.get("type") == "buy" and decision["decision"] == "sell":
                    score += 5  # Sell after buy liquidity has been tapped
                elif pool.get("type") == "sell" and decision["decision"] == "buy":
                    score += 5  # Buy after sell liquidity has been tapped
        
        # 8. Fair Value Gap (FVG) trading - professional Smart Money Concept
        fair_value_gaps = structure_analysis.get("fair_value_gaps", [])
        for gap in fair_value_gaps:
            if gap.get("status") == "unfilled":
                level = gap.get("level", 0)
                size = gap.get("size", 0)
                
                # If we're taking a trade into a fair value gap
                gap_distance = abs(entry_price - level) / entry_price
                if gap_distance < 0.01:  # Within 1% of an unfilled FVG
                    score += 10  # Trading into a fair value gap
                    # Larger FVGs have more significance
                    if size > 5:
                        score += 5  # Large FVG has more significance
            
        # 9. Break and Retest - professional retest trading
        retests = structure_analysis.get("break_and_retest", [])
        for retest in retests:
            if retest.get("status") == "confirmed":
                level = retest.get("level", 0)
                
                # Trading in the direction of a confirmed break and retest
                level_distance = abs(entry_price - level) / entry_price
                if level_distance < 0.01:  # Within 1% of a retest level
                    if retest.get("type") == "support" and decision["decision"] == "buy":
                        score += 15  # Buy at a support retest
                    elif retest.get("type") == "resistance" and decision["decision"] == "sell":
                        score += 15  # Sell at a resistance retest
        
        # 10. Market volatility for superior entries
        if market_data:
            # Calculate recent volatility (standard deviation of percentage changes)
            closes = market_data["closing_prices"][-20:]  # Last 20 closes
            if len(closes) > 1:
                pct_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = sum([abs(x) for x in pct_changes]) / len(pct_changes)
                
                # Calculate avg trading range
                highs = market_data["high_prices"][-20:]
                lows = market_data["low_prices"][-20:]
                avg_range = sum([(highs[i] - lows[i]) / closes[i] for i in range(len(highs))]) / len(highs)
                
                # Ideal volatility for professional traders
                if 0.0015 <= volatility <= 0.004:  # Ideal volatility range
                    score += 12  # Perfect volatility for trading
                elif 0.001 <= volatility < 0.0015:  # Slightly low volatility
                    score += 6
                elif 0.004 < volatility <= 0.008:  # Slightly high volatility
                    score += 6
                elif 0.0005 <= volatility < 0.001:  # Low volatility
                    score -= 5
                elif 0.008 < volatility <= 0.012:  # High volatility
                    score -= 5
                elif volatility > 0.012:  # Extreme volatility
                    score -= 20  # Professional traders avoid extreme volatility
                
                # Trading range analysis
                if 0.004 <= avg_range <= 0.01:  # Ideal range
                    score += 8  # Perfect range for trading
                elif avg_range > 0.01:  # Wide range
                    score += 4  # Good for trend following
                elif avg_range < 0.002:  # Very narrow range
                    score -= 10  # Too tight for good trading
        
        # 11. Confirmation of trade by multiple factors (convergence of signals)
        # Count how many positive factors we have
        positive_factors = 0
        
        # Trend alignment
        if (trend == "uptrend" and decision["decision"] == "buy") or (trend == "downtrend" and decision["decision"] == "sell"):
            positive_factors += 1
            
        # Structure alignment
        if (current_structure == "bullish" and decision["decision"] == "buy") or (current_structure == "bearish" and decision["decision"] == "sell"):
            positive_factors += 1
            
        # Good risk-reward
        if risk_reward >= 2:
            positive_factors += 1
            
        # Strong trend
        if trend_strength == "strong":
            positive_factors += 1
            
        # Trading with the market structure
        if (structure_type == "HH-HL" and decision["decision"] == "buy") or (structure_type == "LH-LL" and decision["decision"] == "sell"):
            positive_factors += 1
            
        # Trading with order blocks
        has_active_order_block = False
        for block in key_order_blocks:
            if block.get("status") == "active" and ((block.get("type") == "bullish" and decision["decision"] == "buy") or 
                                                   (block.get("type") == "bearish" and decision["decision"] == "sell")):
                has_active_order_block = True
                break
        if has_active_order_block:
            positive_factors += 1
            
        # Perfect setup bonus - multiple confirming factors converge
        if positive_factors >= 5:  # Super high-probability setup
            score += 25  # Significant bonus for having multiple aligned factors
        elif positive_factors >= 4:
            score += 15  # Strong convergence of factors
        elif positive_factors >= 3:
            score += 8  # Good convergence
            
        # 12. Final quality adjustments - professional traders are very selective
        # Apply scaling to focus on exceptional setups
        if score < 60:
            score *= 0.85  # Even stronger penalty for mediocre setups
        elif score >= 90:
            score *= 1.2  # Significant boost for exceptional setups
        elif score >= 80:
            score *= 1.1  # Good boost for very good setups
            
        # Return final score
        return score

    async def execute_best_opportunity(self, opportunity):
        """
        Execute the best trading opportunity.
        
        Args:
            opportunity (dict): Information about the best trading opportunity
            
        Returns:
            bool: True if executed successfully, False otherwise
        """
        if not opportunity:
            return False
            
        symbol = opportunity["symbol"]
        decision = opportunity["decision_data"]
        
        self.logger.info(f"Executing best opportunity: {symbol} {decision['decision']} with score {opportunity['score']:.2f}")
        
        # Execute the trade
        trade_executed = await self.execute_trade(symbol, decision)
        
        if trade_executed and trade_executed.get("orderId"):
            self.current_trade = {
                "symbol": symbol,
                "decision": decision["decision"],
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "entry_price": decision["entry_price"],
                "position_size": self.position_size,
                "order_id": trade_executed["orderId"],
                "entry_fee": decision["entry_price"] * self.position_size * self.fee_rate,
                "expected_exit_fee": decision["take_profit"] * self.position_size * self.fee_rate,
                "entry_time": time.time(),  # Add entry time for minimum hold time check
                "opportunity_score": opportunity["score"]  # Store the opportunity score
            }
            # Send Telegram notification about new trade
            await self.notify_trade_opened(symbol, decision)
            return True
        else:
            self.logger.error(f"Failed to execute trade for {symbol}")
            return False

    def _get_min_quantity(self, symbol):
        """
        Get the minimum order quantity for a trading pair from configuration.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Minimum quantity for the symbol
        """
        # Default minimum quantities - expanded list
        default_min_quantities = {
            "BTCUSDT": 0.001,
            "ETHUSDT": 0.01,
            "BNBUSDT": 0.1,
            "XRPUSDT": 10,
            "ADAUSDT": 10,
            "MATICUSDT": 10,
            "DOTUSDT": 1,
            "LINKUSDT": 1,
            "AVAXUSDT": 0.1,
            "TRXUSDT": 100,
            "LTCUSDT": 0.1,
            "UNIUSDT": 0.1,
            "ATOMUSDT": 0.1,
            "ETCUSDT": 0.1,
            "XLMUSDT": 100,
            "NEARUSDT": 1,
            "SHIBUSDT": 100000,
            "ALGOUSDT": 10
        }
        
        # Try to get from config
        if self.config and "trading_pairs" in self.config.get("binance", {}):
            trading_pairs = self.config["binance"]["trading_pairs"]
            for pair in trading_pairs:
                if pair.get("symbol") == symbol and "min_quantity" in pair:
                    return float(pair["min_quantity"])  # Ensure it's a float
        
        # If symbol exists in defaults, return that value
        if symbol in default_min_quantities:
            return default_min_quantities[symbol]
            
        # For unknown symbols, try to intelligently determine a reasonable default
        # Extract the base asset from the symbol
        base_asset = symbol.replace("USDT", "")
        
        # Check if we have any similar base assets in our defaults
        for known_symbol, quantity in default_min_quantities.items():
            known_base = known_symbol.replace("USDT", "")
            if base_asset == known_base:
                self.logger.info(f"Using minimum quantity from similar symbol {known_symbol} for {symbol}")
                return quantity
        
        # Fallback to a conservative default
        self.logger.warning(f"No minimum quantity configured for {symbol}, using default of 0.01")
        return 0.01  # Conservative default for unknown assets

    def _apply_special_handling_for_asset(self, symbol, quantity, latest_price):
        """
        Apply special handling for specific assets that might need adjustments.
        
        Args:
            symbol (str): The trading pair symbol
            quantity (float): The calculated quantity
            latest_price (float): Latest price of the asset
            
        Returns:
            float: Adjusted quantity
        """
        # First check: quantity must be greater than zero
        if quantity <= 0:
            self.logger.warning(f"Special handling: {symbol} quantity is <= 0. Setting to minimum quantity.")
            return self._get_min_quantity(symbol)

        # Special handling for low-value, high-volume assets (need more quantity)
        if symbol in ["DOGEUSDT", "SHIBUSDT", "TRXUSDT", "XLMUSDT"]:
            min_quantity = self._get_min_quantity(symbol)
            if quantity < min_quantity:
                self.logger.info(f"Special handling for {symbol}: Setting to minimum quantity of {min_quantity}")
                quantity = min_quantity
        
        # If actual notional value (quantity * price) is too low, adjust quantity
        min_notional = 100.1  # Default minimum notional value for Binance Futures
        if self.config and 'futures' in self.config.get('binance', {}) and 'min_notional' in self.config['binance']['futures']:
            min_notional = self.config['binance']['futures']['min_notional']
            
        notional = quantity * latest_price
        if notional < min_notional:
            self.logger.info(f"Special handling: Notional value too low ({notional:.2f} USDT < {min_notional} USDT). Adjusting quantity.")
            # Calculate required quantity to meet min notional
            required_quantity = (min_notional * 1.01) / latest_price  # Add 1% buffer
            
            # Ensure we're above the minimum quantity for this asset
            min_quantity = self._get_min_quantity(symbol)
            adjusted_quantity = max(required_quantity, min_quantity)
            
            # Log the adjustment details for troubleshooting
            self.logger.info(f"Adjusted quantity for {symbol}: {quantity} → {adjusted_quantity} (Min notional: {min_notional}, Price: {latest_price})")
            
            return adjusted_quantity
            
        # Return original quantity if no special handling needed
        return quantity