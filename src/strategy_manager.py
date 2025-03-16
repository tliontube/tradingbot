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
        self.min_risk_reward = 1.5  # Default minimum risk-reward ratio
        self.scalp_risk_reward = 2.0  # Default for scalp trades
        
        # Load parameters from config if provided
        if config and 'risk_management' in config:
            risk_config = config['risk_management']
            # Always use 2% of account balance, regardless of config
            self.position_size = 0.02
            self.fee_rate = risk_config.get('fee_rate', 0.0004)
            self.min_profit_multiple = risk_config.get('min_profit_multiple', 3)
            self.min_risk_reward = risk_config.get('min_risk_reward', 1.5)
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

    async def decide_strategy(self, symbol):
        # Check if there is an active trade
        if self.current_trade is not None:
            return {"decision": "waiting", "reason": "Waiting for current trade to complete", "stop_loss": None, "take_profit": None}

        # Fetch market data
        market_data = await self.binance_client.get_market_data(symbol, interval="1m", limit=200)
        if market_data is None:
            return {"decision": "waiting", "reason": "Unable to fetch market data", "stop_loss": None, "take_profit": None}

        # Get trading decision from Gemini
        gemini_decision = self.gemini_client.get_trading_decision(
            closing_prices=market_data["closing_prices"],
            volumes=market_data["volumes"],
            high_prices=market_data["high_prices"],
            low_prices=market_data["low_prices"],
            open_prices=market_data["open_prices"]
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
                # Send Telegram notification about new trade
                await self.notify_trade_opened(symbol, gemini_decision)
            else:
                return {"decision": "waiting", "reason": "Trade execution failed", "stop_loss": None, "take_profit": None}

        return gemini_decision

    async def execute_trade(self, symbol, decision):
        """Execute a trade based on the strategy decision."""
        try:
            # Get account balance to ensure we have enough funds
            balances = await self.binance_client.get_account_balance()
            if not balances:
                self.logger.error("Could not fetch account balance")
                return None

            # Calculate the quantity based on position size
            latest_price = await self.binance_client.get_latest_price(symbol)
            if not latest_price:
                self.logger.error("Could not fetch latest price")
                return None
                
            # Get symbol info to determine the correct precision for quantity
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Could not fetch symbol info for {symbol}")
                # Use default precision if symbol info not available
                quantity_precision = 5
            else:
                # Get the precision for quantity from symbol info
                quantity_precision = symbol_info.get('quantityPrecision', 5)
                
            # For futures trading, calculate quantity in contracts based on leverage
            if self.binance_client.using_futures:
                # Check current position
                position = await self.binance_client.get_position_info(symbol)
                if position:
                    current_position_size = float(position.get("positionAmt", 0))
                    self.logger.info(f"Current position size: {current_position_size}")
                    
                    # If we already have a position in the opposite direction, we need to close it first
                    if (decision["decision"] == "buy" and current_position_size < 0) or \
                       (decision["decision"] == "sell" and current_position_size > 0):
                        self.logger.info(f"Closing existing position before opening new one")
                        close_side = "BUY" if current_position_size < 0 else "SELL"
                        close_quantity = abs(current_position_size)
                        
                        # Determine position side for closing
                        position_params = {}
                        if self.binance_client.futures_position_side == "BOTH":
                            position_side = "SHORT" if current_position_size < 0 else "LONG"
                            position_params["positionSide"] = position_side
                            self.logger.info(f"Closing {position_side} position with {close_side} order")
                        
                        if close_quantity > 0:
                            await self.binance_client.place_order(
                                symbol=symbol,
                                side=close_side,
                                order_type="MARKET",
                                quantity=self._round_to_precision(close_quantity, quantity_precision),
                                reduce_only=True,
                                **position_params
                            )
                
                # Get available balance in USDT
                usdt_balance = balances.get("USDT", {}).get("free", 0)
                
                # Use at least 10 USDT to ensure we can place orders
                min_usdt = 10
                usdt_balance = max(usdt_balance, min_usdt)
                
                # MODIFICATION: Use 2% of the account balance for position sizing
                position_percentage = 0.02  # 2% of account balance
                position_value = usdt_balance * position_percentage
                
                # Ensure minimum position value
                min_position_value = 100.1  # Minimum notional value for Binance Futures is 100 USDT
                
                # Use config value if provided
                if self.config and 'futures' in self.config.get('binance', {}):
                    futures_config = self.config['binance']['futures']
                    min_position_value = futures_config.get('min_notional', 100.1)
                    
                # MODIFICATION: If position value is below minimum, adjust leverage to make it work
                current_leverage = self.binance_client.futures_leverage
                max_allowed_leverage = 125  # Maximum leverage allowed on Binance
                
                if position_value < min_position_value:
                    self.logger.warning(f"Position value ({position_value:.2f} USDT) is below minimum ({min_position_value} USDT).")
                    
                    # Calculate required leverage
                    # Formula: required_leverage = (min_position_value / position_value) * current_leverage
                    required_leverage = math.ceil((min_position_value / position_value) * current_leverage)
                    required_leverage = min(required_leverage, max_allowed_leverage)
                    
                    if required_leverage > current_leverage:
                        self.logger.info(f"Adjusting leverage from {current_leverage}x to {required_leverage}x to enable trading with 2% of balance")
                        
                        # Set new leverage
                        try:
                            response = await self.binance_client.client.futures_change_leverage(
                                symbol=symbol, 
                                leverage=required_leverage
                            )
                            self.binance_client.futures_leverage = required_leverage
                            self.logger.info(f"Successfully adjusted leverage to {response['leverage']}x")
                        except Exception as e:
                            self.logger.error(f"Failed to adjust leverage: {e}")
                            # Continue with current leverage if adjustment fails
                    
                    # Use 2% of account balance regardless, now with adjusted leverage
                    position_value = usdt_balance * position_percentage
                else:
                    self.logger.info(f"Using 2% of account balance ({position_value:.2f} USDT) for position sizing with {current_leverage}x leverage")
                
                # Calculate quantity in BTC (or asset)
                quantity = position_value / latest_price
                
                self.logger.info(f"Futures position: USDT balance={usdt_balance}, position value={position_value}, leverage={self.binance_client.futures_leverage}x, quantity={quantity}")
                
                # Ensure minimum quantity for the specific asset
                # Get minimum quantity for this symbol from config or defaults
                min_quantity = self._get_min_quantity(symbol)
                if quantity < min_quantity:
                    self.logger.warning(f"Calculated quantity ({quantity}) is below minimum ({min_quantity}) for {symbol}. Using minimum quantity.")
                    quantity = min_quantity
            else:
                # For spot trading, use direct quantity based on 2% of balance
                base_asset = symbol.replace("USDT", "")
                base_balance = balances.get(base_asset, {}).get("free", 0)
                
                # Use 2% of the available balance
                quantity = base_balance * 0.02
            
            # Round the quantity to the correct precision
            quantity = self._round_to_precision(quantity, quantity_precision)
            
            # Apply special handling for specific assets
            quantity = self._apply_special_handling_for_asset(symbol, quantity, latest_price)
            
            # Re-round after special handling
            quantity = self._round_to_precision(quantity, quantity_precision)
            
            # Ensure quantity is greater than zero
            if quantity <= 0:
                self.logger.error(f"Calculated quantity is <= 0: {quantity} for {symbol}. Setting to minimum quantity.")
                # Fix for zero quantity - use the minimum quantity for this symbol
                quantity = self._get_min_quantity(symbol)
                # Round again after setting to minimum
                quantity = self._round_to_precision(quantity, quantity_precision)
                # Log the adjusted quantity
                self.logger.info(f"Final quantity after adjustment: {quantity}")
                
                # Double-check if quantity is still zero
                if quantity <= 0:
                    self.logger.error(f"Still unable to calculate a valid quantity for {symbol}. Cannot place order.")
                    return None
            
            self.logger.info(f"Placing {decision['decision']} order for {symbol}: quantity={quantity} at price ~{latest_price}")

            # Place the order with position side for hedged mode
            position_side = None
            if self.binance_client.using_futures and self.binance_client.futures_position_side == "BOTH":
                position_side = "LONG" if decision["decision"] == "buy" else "SHORT"
                self.logger.info(f"Setting position side to {position_side}")
            
            # Place the main entry market order
            entry_order = await self.binance_client.place_order(
                symbol=symbol,
                side=decision["decision"].upper(),
                order_type="MARKET",
                quantity=quantity,
                positionSide=position_side
            )

            if not entry_order:
                self.logger.error("Failed to place entry order")
                return None
                
            # Check if we actually got a successful order or error
            if entry_order.get("status") == "ERROR":
                self.logger.error(f"Error placing entry order: {entry_order.get('msg', 'Unknown error')}")
                return None

            self.logger.info(f"Successfully placed {decision['decision']} order: {entry_order}")
            
            # Place stop loss and take profit orders
            stop_loss_order = None
            take_profit_order = None
            
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
            take_profit_price = self._round_to_precision(decision["take_profit"], price_precision)
            
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
                
                # Try to place take profit order
                try:
                    # Wait a moment between orders
                    await asyncio.sleep(1)
                    
                    # Refresh position info
                    position = await self.binance_client.get_position_info(symbol)
                    actual_quantity = abs(float(position.get("positionAmt", 0))) if position else quantity
                    
                    # If position amount is 0, use the original quantity
                    if actual_quantity <= 0:
                        self.logger.warning(f"Position amount is 0, using original quantity: {quantity}")
                        actual_quantity = quantity
                    
                    # Round to precision to avoid quantity errors
                    actual_quantity = self._round_to_precision(actual_quantity, quantity_precision)
                    
                    # Get position side for hedged mode
                    correct_position_side = None
                    if self.binance_client.futures_position_side == "BOTH":
                        position_amt = float(position.get("positionAmt", 0)) if position else 0
                        correct_position_side = "LONG" if position_amt > 0 else "SHORT"
                        self.logger.info(f"Determined position side for take profit: {correct_position_side} (amount: {position_amt})")
                        
                        # Place take profit order with retry logic
                        for attempt in range(3):  # Max 3 attempts
                            try:
                                if attempt > 0:
                                    self.logger.info(f"Retry #{attempt} for take profit order")
                                    
                                take_profit_params = {
                                    "symbol": symbol,
                                    "side": tp_side,
                                    "order_type": "TAKE_PROFIT_MARKET",
                                    "stop_price": take_profit_price,
                                    "quantity": actual_quantity
                                }
                                
                                # Add position side for hedged mode
                                if correct_position_side:
                                    take_profit_params["positionSide"] = correct_position_side
                                
                                take_profit_order = await self.binance_client.place_order(**take_profit_params)
                                
                                if take_profit_order:
                                    if take_profit_order.get("status") == "ERROR":
                                        self.logger.error(f"Error placing take profit order: {take_profit_order.get('msg', 'Unknown error')}")
                                        if attempt < 2:  # Try again if not the last attempt
                                            await asyncio.sleep(1)
                                            continue
                                    else:
                                        self.logger.info(f"Successfully placed TAKE PROFIT order at {take_profit_price}: {take_profit_order}")
                                        break  # Success, exit retry loop
                                else:
                                    self.logger.error(f"Failed to place take profit order at {take_profit_price}")
                                    if attempt < 2:  # Try again if not the last attempt
                                        await asyncio.sleep(1)
                            except Exception as e:
                                self.logger.error(f"Error on attempt {attempt+1} placing take profit order: {e}")
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
                    self.logger.error(f"Error in take profit order process: {e}")
            
            # Store SL and TP order IDs if they were successfully placed
            if stop_loss_order and stop_loss_order.get("orderId"):
                entry_order["stop_loss_order"] = stop_loss_order
            
            if take_profit_order and take_profit_order.get("orderId"):
                entry_order["take_profit_order"] = take_profit_order
            
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
                    await self.cancel_remaining_orders(symbol)
                
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
            await self.cancel_remaining_orders(symbol)
            
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

    async def cancel_remaining_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            # Check if we have stop loss or take profit order IDs stored
            stop_loss_order_id = self.current_trade.get("stop_loss_order_id")
            take_profit_order_id = self.current_trade.get("take_profit_order_id")
            
            if stop_loss_order_id:
                try:
                    self.logger.info(f"Canceling stop loss order: {stop_loss_order_id}")
                    await self.binance_client.cancel_order(symbol, stop_loss_order_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel stop loss order: {e}")
            
            if take_profit_order_id:
                try:
                    self.logger.info(f"Canceling take profit order: {take_profit_order_id}")
                    await self.binance_client.cancel_order(symbol, take_profit_order_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel take profit order: {e}")
                    
            # As a fallback, also try to cancel all open orders for this symbol
            try:
                self.logger.info(f"Canceling all open orders for {symbol}")
                await self.binance_client.cancel_all_orders(symbol)
            except Exception as e:
                self.logger.warning(f"Error canceling all orders: {e}")
                
            return True
        except Exception as e:
            self.logger.error(f"Error canceling remaining orders: {e}")
            return False

    async def notify_trade_opened(self, symbol, decision):
        """Send notification when a new trade is opened."""
        message = (
            f"🚨 *New Trade Opened* 🚨\n\n"
            f"📊 **Symbol**: {symbol}\n"
            f"🔄 **Action**: {decision['decision'].upper()}\n"
            f"📝 **Reason**: {decision['reason']}\n"
            f"💰 **Entry Price**: {decision['entry_price']}\n"
            f"🛑 **Stop-Loss**: {decision['stop_loss']} " + 
            (f"(automatic order placed)" if self.current_trade.get("stop_loss_order_id") else "(manual monitoring)") + 
            f"\n"
            f"🎯 **Take-Profit**: {decision['take_profit']} " + 
            (f"(automatic order placed)" if self.current_trade.get("take_profit_order_id") else "(manual monitoring)") + 
            f"\n"
            f"📈 **Risk-Reward Ratio**: {((decision['take_profit'] - decision['entry_price']) / (decision['entry_price'] - decision['stop_loss'])):.2f}\n\n"
            f"Trading started! 🚀"
        )
        await self.telegram_bot.send_message(message)

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
        Analyze multiple trading pairs and find the best opportunity.
        
        Args:
            trading_pairs (list): List of trading pair symbols to analyze
            
        Returns:
            dict: Information about the best trading opportunity
        """
        self.logger.info(f"Scanning {len(trading_pairs)} trading pairs for opportunities...")
        
        opportunities = []
        
        # Analyze each trading pair
        for symbol in trading_pairs:
            self.logger.info(f"Analyzing {symbol}...")
            
            try:
                # Fetch market data
                market_data = await self.binance_client.get_market_data(symbol, interval="1m", limit=200)
                if market_data is None:
                    self.logger.warning(f"Unable to fetch market data for {symbol}")
                    continue

                # Get trading decision from Gemini
                gemini_decision = self.gemini_client.get_trading_decision(
                    closing_prices=market_data["closing_prices"],
                    volumes=market_data["volumes"],
                    high_prices=market_data["high_prices"],
                    low_prices=market_data["low_prices"],
                    open_prices=market_data["open_prices"],
                    symbol=symbol  # Pass symbol to Gemini for context
                )
                
                if gemini_decision is None:
                    self.logger.warning(f"Unable to fetch trading decision for {symbol}")
                    continue

                # Skip if the decision is to wait
                if gemini_decision["decision"] == "waiting":
                    continue
                    
                # Calculate opportunity score
                score = self._calculate_opportunity_score(symbol, gemini_decision, market_data)
                
                # Add to opportunities list
                opportunities.append({
                    "symbol": symbol,
                    "decision": gemini_decision["decision"],
                    "decision_data": gemini_decision,
                    "score": score,
                    "market_data": market_data
                })
                
                self.logger.info(f"{symbol} opportunity score: {score:.2f}, decision: {gemini_decision['decision']}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Find the best opportunity
        if not opportunities:
            self.logger.info("No trading opportunities found")
            return None
            
        # Sort opportunities by score (highest first)
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        best_opportunity = opportunities[0]
        
        self.logger.info(f"Best opportunity: {best_opportunity['symbol']} with score {best_opportunity['score']:.2f}")
        return best_opportunity

    def _calculate_opportunity_score(self, symbol, decision, market_data):
        """
        Calculate a score for a trading opportunity based on various factors.
        
        Args:
            symbol (str): The trading pair symbol
            decision (dict): The trading decision
            market_data (dict): Market data for the symbol
            
        Returns:
            float: Opportunity score (higher is better)
        """
        # Initialize base score
        score = 50.0
        
        # Skip if not a buy or sell decision
        if decision["decision"] not in ["buy", "sell"]:
            return 0.0
            
        # Extract necessary data
        entry_price = decision["entry_price"]
        stop_loss = decision["stop_loss"]
        take_profit = decision["take_profit"]
        trend = decision.get("trend", "unknown")
        trend_strength = decision.get("trend_strength", "unknown")
        
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
        
        # Factors affecting score
        
        # 1. Trend alignment (higher score for strong trends)
        if trend == "uptrend" and decision["decision"] == "buy":
            score += 10
        elif trend == "downtrend" and decision["decision"] == "sell":
            score += 10
        elif trend == "ranging":
            score -= 5
            
        # 2. Trend strength
        if trend_strength == "strong":
            score += 10
        elif trend_strength == "moderate":
            score += 5
        elif trend_strength == "weak":
            score -= 5
            
        # 3. Risk-reward ratio (higher is better)
        if risk_reward >= 3:
            score += 20
        elif risk_reward >= 2:
            score += 10
        elif risk_reward >= 1.5:
            score += 5
        else:
            score -= 10
            
        # 4. Profit potential relative to fees
        profit_multiple = potential_profit / round_trip_fee if round_trip_fee > 0 else 0
        if profit_multiple >= 5:
            score += 15
        elif profit_multiple >= 3:
            score += 10
        elif profit_multiple >= 2:
            score += 5
        else:
            score -= 5
            
        # 5. Market volatility (from market data)
        if market_data:
            # Calculate recent volatility (standard deviation of percentage changes)
            closes = market_data["closing_prices"][-20:]  # Last 20 closes
            if len(closes) > 1:
                pct_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = sum([abs(change) for change in pct_changes]) / len(pct_changes)
                
                # Moderate volatility is good, extreme is bad
                if 0.005 <= volatility <= 0.015:
                    score += 10  # Good volatility range
                elif volatility > 0.03:
                    score -= 10  # Too volatile
                    
        # 6. Structure Analysis Bonus
        if "structure_analysis" in decision:
            structure = decision["structure_analysis"].get("current_structure", "")
            
            # Bonus for bullish structure on buy signals
            if decision["decision"] == "buy" and "bullish" in structure:
                score += 15
                
            # Bonus for bearish structure on sell signals
            elif decision["decision"] == "sell" and "bearish" in structure:
                score += 15
                
        # 7. Volume Analysis
        if market_data and "volumes" in market_data:
            volumes = market_data["volumes"][-10:]  # Last 10 volumes
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            last_volume = volumes[-1] if volumes else 0
            
            # Higher volume supporting the move is good
            if last_volume > avg_volume * 1.5:
                score += 10
                
        # Ensure score stays in a reasonable range
        score = max(0, min(score, 100))
        
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
                "stop_loss_order_id": trade_executed.get("stop_loss_order", {}).get("orderId"),
                "take_profit_order_id": trade_executed.get("take_profit_order", {}).get("orderId"),
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
            "SOLUSDT": 0.1,
            "DOGEUSDT": 100,
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