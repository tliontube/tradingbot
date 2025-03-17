import asyncio
import math

class TradeExecutor:
    def __init__(self, binance_client, config):
        self.binance_client = binance_client
        self.config = config
        self.active_trailing_stops = {}  # Track active trailing stops
        self.max_concurrent_trades = 3  # Add maximum concurrent trades limit

    async def _adjust_leverage_for_trade(self, symbol, required_notional):
        """Adjust leverage based on available balance and required position size."""
        try:
            # Get account balance
            balances = await self.binance_client.get_account_balance()
            if not balances or "USDT" not in balances:
                return False
            
            available_balance = float(balances["USDT"]["free"])
            current_leverage = await self.binance_client.get_leverage(symbol)
            
            # Calculate required leverage
            required_leverage = math.ceil(required_notional / available_balance)
            
            # Cap leverage at 50x for safety
            new_leverage = min(max(required_leverage, current_leverage), 50)
            
            if new_leverage != current_leverage:
                await self.binance_client.set_leverage(symbol, new_leverage)
                self.binance_client.logger.info(f"Adjusted leverage for {symbol} from {current_leverage}x to {new_leverage}x")
            
            return True
        except Exception as e:
            self.binance_client.logger.error(f"Error adjusting leverage: {e}")
            return False

    async def _calculate_position_size(self, symbol, latest_price, available_balance):
        """Calculate position size based on 5% of account balance."""
        try:
            # Use 5% of available balance
            position_amount = available_balance * 0.05
            # Calculate base quantity
            base_quantity = position_amount / latest_price
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            quantity_precision = symbol_info.get('quantityPrecision', 3)
            # Calculate minimum quantity for 100.1 USDT notional
            min_notional = 100.1  # Minimum notional value required
            min_quantity = min_notional / latest_price
            
            # Use the larger of calculated quantity or minimum required
            quantity = max(base_quantity, min_quantity)
            # Calculate required leverage
            required_leverage = math.ceil((quantity * latest_price) / position_amount)
            
            # Cap leverage at 50x for safety (changed from 15x)
            new_leverage = min(required_leverage, 50)
            
            # Set the leverage
            await self.binance_client.set_leverage(symbol, new_leverage)
            
            # Round quantity to correct precision
            quantity = math.floor(quantity * (10 ** quantity_precision)) / (10 ** quantity_precision)
            
            # Verify notional value meets minimum requirement
            notional = quantity * latest_price
            if notional < min_notional:
                # Increase quantity to meet minimum notional
                quantity = math.ceil(min_notional / latest_price * (10 ** quantity_precision)) / (10 ** quantity_precision)
            
            return quantity, new_leverage
        except Exception as e:
            self.binance_client.logger.error(f"Error calculating position size: {e}")
            return None, None

    async def execute_trade(self, symbol, decision):
        """Execute a trade based on the strategy decision."""
        # Get symbol information and latest price
        symbol_info = await self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            return None

        latest_price = await self.binance_client.get_latest_price(symbol)
        if not latest_price:
            return None

        # Get account balance
        balances = await self.binance_client.get_account_balance()
        if not balances or "USDT" not in balances:
            return None
        
        available_balance = float(balances["USDT"]["free"])
        
        # Calculate position size and set leverage
        quantity, new_leverage = await self._calculate_position_size(symbol, latest_price, available_balance)
        if not quantity or not new_leverage:
            return None

        # Extract order side and determine position side
        order_side = decision["decision"].upper()
        position_side = "LONG" if order_side == "BUY" else "SHORT"
        
        # Place the entry order
        entry_order = await self.binance_client.place_order(
            symbol=symbol,
            side=order_side,
            order_type="MARKET",
            quantity=quantity,
            positionSide=position_side
        )
        if not entry_order or entry_order.get("status") == "ERROR":
            return None

        # Store stop-loss and take-profit internally
        decision["quantity"] = quantity
        decision["position_side"] = position_side
        self.active_trailing_stops[symbol] = {
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"],
            "entry_price": decision["entry_price"],
            "quantity": quantity,
            "position_side": position_side
        }

        return {
            "entry_order": entry_order,
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"]
        }

    async def _place_stop_order(self, symbol, side, order_type, quantity, stop_price, position_side):
        """Helper method to place stop loss or take profit orders."""
        try:
            order = await self.binance_client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                stop_price=stop_price,
                positionSide=position_side,
                reduce_only=True  # Add directly in the place_order call
            )
            if not order or order.get("status") == "ERROR":
                self.binance_client.logger.error(f"Failed to place {order_type} order for {symbol}: {order}")
                return None
            return order
        except Exception as e:
            self.binance_client.logger.error(f"Error placing {order_type} order for {symbol}: {e}")
            return None

    async def _update_trailing_stop(self, symbol, position_side, current_price, initial_stop_price):
        """Update trailing stop based on price movement."""
        try:
            if symbol not in self.active_trailing_stops:
                self.active_trailing_stops[symbol] = {
                    'initial_stop': initial_stop_price,
                    'current_stop': initial_stop_price,
                    'activation_price': current_price * (1 + self.config['risk_management']['trailing_stop_activation'] if position_side == "LONG" else -self.config['risk_management']['trailing_stop_activation'])
                }
                return initial_stop_price

            trailing_data = self.active_trailing_stops[symbol]
            trailing_percent = self.config['risk_management']['trailing_stop_percent']

            if position_side == "LONG":
                # For long positions
                if current_price >= trailing_data['activation_price']:
                    new_stop = current_price * (1 - trailing_percent)
                    if new_stop > trailing_data['current_stop']:
                        trailing_data['current_stop'] = new_stop
                        # Update the stop loss order
                        await self._modify_stop_loss(symbol, new_stop, position_side)
                        self.binance_client.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
            else:
                # For short positions
                if current_price <= trailing_data['activation_price']:
                    new_stop = current_price * (1 + trailing_percent)
                    if new_stop < trailing_data['current_stop']:
                        trailing_data['current_stop'] = new_stop
                        # Update the stop loss order
                        await self._modify_stop_loss(symbol, new_stop, position_side)
                        self.binance_client.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")

            return trailing_data['current_stop']
        except Exception as e:
            self.binance_client.logger.error(f"Error updating trailing stop: {e}")
            return initial_stop_price

    async def _modify_stop_loss(self, symbol, new_stop_price, position_side):
        """Modify existing stop loss order."""
        try:
            # Cancel existing stop loss
            await self.binance_client.cancel_all_orders(symbol)
            
            # Get position details
            position = await self.binance_client.get_position_info(symbol)
            if not position:
                return False
                
            quantity = abs(float(position.get("positionAmt", 0)))
            if quantity == 0:
                return False

            # Place new stop loss order
            order_side = "SELL" if position_side == "LONG" else "BUY"
            await self._place_stop_order(
                symbol=symbol,
                side=order_side,
                order_type="STOP_MARKET",
                quantity=quantity,
                stop_price=new_stop_price,
                position_side=position_side
            )
            return True
        except Exception as e:
            self.binance_client.logger.error(f"Error modifying stop loss: {e}")
            return False

    async def check_trade_completion(self, symbol, current_trade, notification_manager):
        """Check if the current trade is complete and handle stop-loss/take-profit internally."""
        try:
            # Fetch the latest price
            latest_price = await self.binance_client.get_latest_price(symbol)
            if not latest_price:
                self.binance_client.logger.warning(f"Could not fetch latest price for {symbol}.")
                return False

            # Retrieve internal stop-loss and take-profit levels
            trade_data = self.active_trailing_stops.get(symbol)
            if not trade_data:
                self.binance_client.logger.warning(f"No active trade data found for {symbol}.")
                return False

            stop_loss = trade_data["stop_loss"]
            take_profit = trade_data["take_profit"]
            position_side = trade_data["position_side"]

            # Check if stop-loss or take-profit conditions are met
            if ((position_side == "LONG" and latest_price <= stop_loss) or
                (position_side == "SHORT" and latest_price >= stop_loss)):
                result = "Loss"
                exit_price = stop_loss
            elif ((position_side == "LONG" and latest_price >= take_profit) or
                  (position_side == "SHORT" and latest_price <= take_profit)):
                result = "Win"
                exit_price = take_profit
            else:
                return False  # Trade is still active

            # Close the position
            position_closed = await self.close_position(symbol)
            if not position_closed:
                self.binance_client.logger.warning(f"Failed to close position for {symbol}.")
                return False

            # Notify about trade closure
            await notification_manager.notify_trade_closed(
                symbol,
                result,
                trade_data["entry_price"],
                exit_price
            )

            # Remove the trade from active tracking
            del self.active_trailing_stops[symbol]
            return True

        except Exception as e:
            self.binance_client.logger.error(f"Error checking trade completion for {symbol}: {e}")
            return False

    async def close_position(self, symbol):
        """Close the open position for the given symbol."""
        try:
            # Get the current position information
            position = await self.binance_client.get_position_info(symbol)
            if not position or float(position.get("positionAmt", 0)) == 0:
                self.binance_client.logger.info(f"No active position found for {symbol}.")
                return True

            # Cancel all existing orders first
            self.binance_client.logger.info(f"Cancelling all open orders for {symbol}...")
            await self.binance_client.cancel_all_orders(symbol)

            # Get position details
            position_amt = float(position["positionAmt"])
            position_side = position.get("positionSide", "BOTH")

            # Determine the side to close the position
            close_side = "SELL" if position_amt > 0 else "BUY"
            quantity = abs(position_amt)

            # Round the quantity to the correct precision
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            quantity_precision = symbol_info.get("quantityPrecision", 6) if symbol_info else 6
            quantity = round(quantity, quantity_precision)

            # Ensure quantity is valid
            if quantity <= 0:
                self.binance_client.logger.warning(f"Invalid position quantity for {symbol}: {quantity}")
                return False

            # Place a market order to close the position
            for attempt in range(3):  # Retry up to 3 times
                try:
                    self.binance_client.logger.info(f"Attempting to close position for {symbol}, quantity: {quantity}...")
                    close_order = await self.binance_client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="MARKET",
                        quantity=quantity,
                        positionSide=position_side,
                        reduce_only=True
                    )

                    if close_order and close_order.get("status") != "ERROR":
                        # Verify position is closed
                        await asyncio.sleep(1)  # Wait for position to update
                        new_position = await self.binance_client.get_position_info(symbol)
                        if new_position and abs(float(new_position.get("positionAmt", 0))) < 0.000001:
                            self.binance_client.logger.info(f"Position successfully closed for {symbol}.")
                            return True
                except Exception as e:
                    self.binance_client.logger.warning(f"Close position attempt {attempt + 1} failed for {symbol}: {e}")
                    await asyncio.sleep(1)

            self.binance_client.logger.error(f"Failed to close position for {symbol} after 3 attempts.")
            return False
        except Exception as e:
            self.binance_client.logger.error(f"Error closing position for {symbol}: {e}")
            return False
