#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import yaml
import json
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_client import BinanceClient
from gemini_client import GeminiClient
from telegram_bot import TelegramBot
from strategy_manager import StrategyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only
    ]
)
logger = logging.getLogger("trade_simulator")

class MockBinanceClient:
    """Mock BinanceClient for testing without actual API calls."""
    
    def __init__(self):
        self.using_futures = True
        self.futures_position_side = "BOTH"
        self.futures_leverage = 10
        self.orders = []
        self.mock_price = 50000.0  # Starting price for BTC
        self.positions = {}
        self.stop_losses = []
        self.take_profits = []
        self.price_precision = 2
        self.quantity_precision = 6
        
    async def get_account_balance(self):
        return {
            "USDT": {"free": 10000.0, "locked": 0.0, "total": 10000.0},
            "BTC": {"free": 0.1, "locked": 0.0, "total": 0.1}
        }
        
    async def get_market_data(self, symbol, interval="5m", limit=500):
        # Generate mock market data
        close_prices = [self.mock_price - i * 10 for i in range(limit)]
        close_prices.reverse()  # Oldest first
        
        # Add a trend to simulate a real market
        for i in range(1, limit):
            close_prices[i] = close_prices[i-1] * (1 + (0.0001 * (i % 5 - 2)))
            
        # Most recent price is our current mock price
        close_prices[-1] = self.mock_price
        
        high_prices = [p * 1.005 for p in close_prices]
        low_prices = [p * 0.995 for p in close_prices]
        open_prices = [p * 0.998 for p in close_prices]
        volumes = [100 + i % 10 for i in range(limit)]
        
        return {
            "close": close_prices,
            "volume": volumes,
            "high": high_prices,
            "low": low_prices,
            "open": open_prices
        }
        
    async def get_symbol_info(self, symbol):
        return {
            "symbol": symbol,
            "pricePrecision": self.price_precision,
            "quantityPrecision": self.quantity_precision,
            "filters": {
                "LOT_SIZE": {
                    "minQty": "0.000001",
                    "maxQty": "9000",
                    "stepSize": "0.000001"
                }
            }
        }
        
    async def get_latest_price(self, symbol):
        return self.mock_price
        
    async def get_leverage(self, symbol):
        return self.futures_leverage
        
    async def place_order(self, symbol, side, order_type, quantity, **kwargs):
        order_id = len(self.orders) + 1
        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "quantity": quantity,
            "status": "FILLED",
            "avgPrice": self.mock_price,
            **kwargs
        }
        
        # Track our position when a market order gets filled
        if order_type == "MARKET":
            position_side = kwargs.get("positionSide", "LONG" if side == "BUY" else "SHORT")
            
            # Initialize position if not exists
            if symbol not in self.positions:
                self.positions[symbol] = {"positionAmt": 0, "entryPrice": self.mock_price, "positionSide": None}
            
            # Update position
            amt = float(quantity)
            if side == "SELL":
                amt = -amt
                
            if position_side == "SHORT":
                amt = -amt
                
            self.positions[symbol]["positionAmt"] += amt
            self.positions[symbol]["positionSide"] = position_side
            
            logger.info(f"Position updated for {symbol}: {self.positions[symbol]}")
            
        # Track stop loss orders
        if order_type == "STOP_MARKET":
            self.stop_losses.append(order)
            logger.info(f"Stop loss order placed: {order}")
            
        # Track take profit orders
        if order_type == "TAKE_PROFIT_MARKET":
            self.take_profits.append(order)
            logger.info(f"Take profit order placed: {order}")
            
        self.orders.append(order)
        return order
        
    async def get_position_info(self, symbol):
        if symbol in self.positions:
            return self.positions[symbol]
        return None
        
    async def get_open_orders(self, symbol):
        return [order for order in self.orders if order["symbol"] == symbol and order.get("status") != "FILLED"]
        
    async def cancel_order(self, symbol, order_id):
        for order in self.orders:
            if order["symbol"] == symbol and order["orderId"] == order_id:
                order["status"] = "CANCELED"
                logger.info(f"Order canceled: {order}")
                return {"status": "SUCCESS"}
        return {"status": "ERROR", "msg": "Order not found"}
        
    async def cancel_all_open_orders(self, symbol):
        for order in self.orders:
            if order["symbol"] == symbol and order.get("status") != "FILLED":
                order["status"] = "CANCELED"
        return {"status": "SUCCESS"}
        
    def get_price_precision(self, symbol):
        return self.price_precision
        
    def simulate_price_move(self, symbol, percent_move):
        """Simulate a price movement and check if it would trigger any orders."""
        old_price = self.mock_price
        self.mock_price = old_price * (1 + percent_move)
        logger.info(f"Price moved from {old_price} to {self.mock_price} ({percent_move*100:+.2f}%)")
        
        # Check if price would trigger any take profit orders
        triggered_tps = []
        for tp in self.take_profits:
            if tp["symbol"] == symbol and tp.get("status") != "FILLED":
                stop_price = float(tp.get("stop_price", 0))
                
                # For buy positions (long), price needs to go above take profit
                if self.positions[symbol]["positionSide"] == "LONG":
                    if self.mock_price >= stop_price:
                        tp["status"] = "FILLED"
                        triggered_tps.append(tp)
                        logger.info(f"TAKE PROFIT TRIGGERED at {stop_price}")
                        
                # For sell positions (short), price needs to go below take profit
                elif self.positions[symbol]["positionSide"] == "SHORT":
                    if self.mock_price <= stop_price:
                        tp["status"] = "FILLED"
                        triggered_tps.append(tp)
                        logger.info(f"TAKE PROFIT TRIGGERED at {stop_price}")
        
        # Check if price would trigger any stop loss orders
        triggered_sls = []
        for sl in self.stop_losses:
            if sl["symbol"] == symbol and sl.get("status") != "FILLED":
                stop_price = float(sl.get("stop_price", 0))
                
                # For buy positions (long), price needs to go below stop loss
                if self.positions[symbol]["positionSide"] == "LONG":
                    if self.mock_price <= stop_price:
                        sl["status"] = "FILLED"
                        triggered_sls.append(sl)
                        logger.info(f"STOP LOSS TRIGGERED at {stop_price}")
                        
                # For sell positions (short), price needs to go above stop loss
                elif self.positions[symbol]["positionSide"] == "SHORT":
                    if self.mock_price >= stop_price:
                        sl["status"] = "FILLED"
                        triggered_sls.append(sl)
                        logger.info(f"STOP LOSS TRIGGERED at {stop_price}")
                        
        # Reduce position if take profit or stop loss was triggered
        position_change = 0
        for order in triggered_tps + triggered_sls:
            amt = float(order["quantity"])
            if self.positions[symbol]["positionAmt"] > 0:  # Long position
                position_change -= amt
            else:  # Short position
                position_change += amt
                
        if position_change != 0:
            old_amt = self.positions[symbol]["positionAmt"]
            self.positions[symbol]["positionAmt"] += position_change
            logger.info(f"Position amount changed from {old_amt} to {self.positions[symbol]['positionAmt']}")
            
            # If this was a take profit order being triggered, log additional information
            if triggered_tps:
                for tp in triggered_tps:
                    profit_percent = ((float(tp.get("stop_price", 0)) / self.positions[symbol]["entryPrice"]) - 1) * 100
                    logger.info(f"Take profit filled: Quantity {tp['quantity']}, Profit: {profit_percent:.2f}%")
        
        return triggered_tps, triggered_sls


class MockGeminiClient:
    """Mock GeminiClient for testing without actual API calls."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    async def get_trading_decision(self, closing_prices, volumes, high_prices, low_prices, open_prices, symbol="BTCUSDT", lookback_period=200):
        """Return a mock trading decision with multiple take-profit levels."""
        
        # Get the latest price from the closing prices
        current_price = closing_prices[-1]
        
        # Create a mock trading decision
        decision = {
            "decision": "buy",
            "reason": "Mock trading decision for testing multiple take-profit levels",
            "confidence": 8,
            "entry_price": current_price,
            "stop_loss": current_price * 0.97,  # 3% below entry
            "take_profit": current_price * 1.06,  # 6% above entry (TP2)
            "take_profit_levels": {
                "tp1": current_price * 1.03,  # 3% above entry
                "tp2": current_price * 1.06,  # 6% above entry
                "tp3": current_price * 1.09   # 9% above entry
            },
            "trend": "uptrend",
            "trend_strength": "strong",
            "risk_reward_ratio": 2.0,
            "structure_analysis": {
                "current_structure": "bullish",
                "last_bos_level": current_price * 0.95,
                "key_order_blocks": [
                    {
                        "type": "bullish",
                        "price": current_price * 0.96,
                        "status": "active"
                    }
                ],
                "liquidity_pools": [
                    {
                        "level": current_price * 0.94,
                        "type": "buy side",
                        "status": "untapped"
                    }
                ],
                "market_structure": {
                    "recent_highs": [current_price * 1.02, current_price * 1.01],
                    "recent_lows": [current_price * 0.98, current_price * 0.99],
                    "structure_type": "HH-HL",
                    "last_choch": current_price * 0.97,
                    "prev_higher_high": current_price * 1.02,
                    "prev_lower_high": current_price * 1.01,
                    "next_key_level": current_price * 1.05
                },
                "fair_value_gaps": [
                    {
                        "level": current_price * 1.02,
                        "size": 0.5,
                        "status": "unfilled"
                    }
                ],
                "break_and_retest": [
                    {
                        "level": current_price * 0.96,
                        "type": "support",
                        "status": "confirmed"
                    }
                ],
                "wyckoff_phase": "accumulation",
                "volume_analysis": "confirming"
            }
        }
        
        return decision


class MockTelegramBot:
    """Mock TelegramBot for testing without actual Telegram messages."""
    
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.messages = []
        
    async def send_message(self, message):
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
        logger.info(f"Telegram message: {message[:100]}...")


async def load_config():
    """Load configuration file."""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return a default configuration for testing
        return {
            "binance": {
                "api_key": "mock_api_key",
                "api_secret": "mock_api_secret",
                "testnet": True,
                "trading_pairs": ["BTCUSDT"],
                "fee_rate": 0.0004,
                "futures": {
                    "enabled": True,
                    "leverage": 10,
                    "margin_type": "ISOLATED"
                }
            },
            "risk_management": {
                "position_size": 0.02,
                "fee_rate": 0.0004,
                "min_profit_multiple": 3,
                "min_risk_reward": 1.5,
                "trailing_stop_percent": 0.005,
                "trailing_stop_activation": 0.01,
                "min_hold_time_seconds": 0  # No minimum hold time for testing
            },
            "telegram": {
                "API_TOKEN": "mock_token",
                "CHAT_ID": "mock_chat_id",
                "enabled": True
            },
            "gemini": {
                "api_key": "mock_gemini_key"
            }
        }


async def simulate_trading():
    """Simulate the trading process."""
    print("=" * 80)
    print("STARTING TRADING SIMULATION")
    print("=" * 80)
    logger.info("Starting trading simulation...")
    
    # Load configuration
    config = await load_config()
    
    # Create mock clients
    binance_client = MockBinanceClient()
    gemini_client = MockGeminiClient(config["gemini"]["api_key"])
    telegram_bot = MockTelegramBot(
        config["telegram"].get("API_TOKEN", "mock_token"),
        config["telegram"].get("CHAT_ID", "mock_chat_id")
    )
    
    # Create strategy manager
    strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)
    
    # Force min_hold_time_seconds to 0 for testing
    strategy_manager.min_hold_time_seconds = 0
    logger.info(f"Setting min_hold_time_seconds to 0 for testing")
    
    # Set the symbol to test
    symbol = "BTCUSDT"
    
    # Step 1: Get market data
    print("\n" + "=" * 50)
    print("STEP 1: FETCHING MARKET DATA")
    print("=" * 50)
    logger.info("Fetching market data...")
    market_data = await binance_client.get_market_data(symbol)
    logger.info(f"Current price: {market_data['close'][-1]}")
    
    # Step 2: Get trading decision
    print("\n" + "=" * 50)
    print("STEP 2: GETTING TRADING DECISION")
    print("=" * 50)
    logger.info("Getting trading decision...")
    decision = await gemini_client.get_trading_decision(
        market_data["close"], 
        market_data["volume"],
        market_data["high"],
        market_data["low"],
        market_data["open"],
        symbol
    )
    
    logger.info(f"Decision: {decision['decision']} at {decision['entry_price']}")
    logger.info(f"Take profit levels: {decision['take_profit_levels']}")
    
    # Step 3: Process and execute the trading decision
    print("\n" + "=" * 50)
    print("STEP 3: PROCESSING TRADING DECISION")
    print("=" * 50)
    logger.info("Processing trading decision...")
    result = await strategy_manager.process_trading_decision(symbol, decision)
    
    if result["decision"] != "waiting":
        logger.info("Trade executed!")
        
        # Step 4: Check the orders that were placed
        print("\n" + "=" * 50)
        print("STEP 4: CHECKING PLACED ORDERS")
        print("=" * 50)
        tp_count = len(binance_client.take_profits)
        logger.info(f"Take profit orders placed: {tp_count}")
        for i, tp in enumerate(binance_client.take_profits):
            logger.info(f"TP{i+1}: Price: {tp['stop_price']}, Quantity: {tp['quantity']}")
        
        sl_count = len(binance_client.stop_losses)
        logger.info(f"Stop loss orders placed: {sl_count}")
        for sl in binance_client.stop_losses:
            logger.info(f"SL: Price: {sl['stop_price']}, Quantity: {sl['quantity']}")
        
        # Step 5: Simulate price movements to test trailing stop loss and take profits
        print("\n" + "=" * 50)
        print("STEP 5: SIMULATING PRICE MOVEMENTS")
        print("=" * 50)
        logger.info("\n===== SIMULATING PRICE MOVEMENTS =====")
        
        # Small price movement (not enough to trigger trailing stop)
        print("\n" + "-" * 40)
        print("Small price movement (0.5%)")
        print("-" * 40)
        logger.info("\n----- Small price movement (0.5%) -----")
        await asyncio.sleep(1)
        binance_client.simulate_price_move(symbol, 0.005)  # 0.5% up
        await strategy_manager.check_trade_completion(symbol)
        
        # Price movement above trailing stop activation threshold
        print("\n" + "-" * 40)
        print("Price movement above trailing stop activation (1.5%)")
        print("-" * 40)
        logger.info("\n----- Price movement above trailing stop activation (1.5%) -----")
        await asyncio.sleep(1)
        binance_client.simulate_price_move(symbol, 0.01)  # 1% up
        await strategy_manager.check_trade_completion(symbol)
        
        # Log current stop loss after trailing stop adjustment
        logger.info(f"Current stop loss: {strategy_manager.current_trade['stop_loss']}")
        
        # Further price movement to test trailing stop updates
        print("\n" + "-" * 40)
        print("Further price movement (2%)")
        print("-" * 40)
        logger.info("\n----- Further price movement (2%) -----")
        await asyncio.sleep(1)
        binance_client.simulate_price_move(symbol, 0.02)  # 2% up
        await strategy_manager.check_trade_completion(symbol)
        
        # Log current stop loss after trailing stop adjustment
        logger.info(f"Current stop loss: {strategy_manager.current_trade['stop_loss']}")
        
        # Price movement to trigger first take profit level
        print("\n" + "-" * 40)
        print("Price movement to TP1")
        print("-" * 40)
        logger.info("\n----- Price movement to TP1 -----")
        tp1_price = decision['take_profit_levels']['tp1']
        current_price = binance_client.mock_price
        percent_to_tp1 = (tp1_price / current_price) - 1
        
        await asyncio.sleep(1)
        tps, sls = binance_client.simulate_price_move(symbol, percent_to_tp1)
        await strategy_manager.check_trade_completion(symbol)
        
        if tps:
            logger.info(f"TP1 triggered! Orders filled: {len(tps)}")
        
        # Price movement to trigger second take profit level
        print("\n" + "-" * 40)
        print("Price movement to TP2")
        print("-" * 40)
        logger.info("\n----- Price movement to TP2 -----")
        tp2_price = decision['take_profit_levels']['tp2']
        current_price = binance_client.mock_price
        percent_to_tp2 = (tp2_price / current_price) - 1
        
        await asyncio.sleep(1)
        tps, sls = binance_client.simulate_price_move(symbol, percent_to_tp2)
        await strategy_manager.check_trade_completion(symbol)
        
        if tps:
            logger.info(f"TP2 triggered! Orders filled: {len(tps)}")
        
        # Price movement to trigger third take profit level
        print("\n" + "-" * 40)
        print("Price movement to TP3")
        print("-" * 40)
        logger.info("\n----- Price movement to TP3 -----")
        tp3_price = decision['take_profit_levels']['tp3']
        current_price = binance_client.mock_price
        percent_to_tp3 = (tp3_price / current_price) - 1
        
        await asyncio.sleep(1)
        tps, sls = binance_client.simulate_price_move(symbol, percent_to_tp3)
        await strategy_manager.check_trade_completion(symbol)
        
        if tps:
            logger.info(f"TP3 triggered! Orders filled: {len(tps)}")
        
        # Check if any orders are still open
        open_orders = await binance_client.get_open_orders(symbol)
        logger.info(f"Remaining open orders: {len(open_orders)}")
        
        # Simulate a different scenario where price drops to trigger stop loss
        print("\n" + "=" * 50)
        print("STOP LOSS SCENARIO SIMULATION")
        print("=" * 50)
        logger.info("\n===== SIMULATING STOP LOSS SCENARIO =====")
        
        # Reset the mock environment
        binance_client = MockBinanceClient()
        gemini_client = MockGeminiClient(config["gemini"]["api_key"])
        telegram_bot = MockTelegramBot(
            config["telegram"].get("API_TOKEN", "mock_token"),
            config["telegram"].get("CHAT_ID", "mock_chat_id")
        )
        
        # Create a new strategy manager
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)
        
        # Force min_hold_time_seconds to 0 for testing
        strategy_manager.min_hold_time_seconds = 0
        logger.info(f"Setting min_hold_time_seconds to 0 for testing")
        
        # Get new market data and decision
        market_data = await binance_client.get_market_data(symbol)
        decision = await gemini_client.get_trading_decision(
            market_data["close"], 
            market_data["volume"],
            market_data["high"],
            market_data["low"],
            market_data["open"],
            symbol
        )
        
        # Execute the trade
        await strategy_manager.process_trading_decision(symbol, decision)
        
        # Price drops to trigger stop loss
        print("\n" + "-" * 40)
        print("Price drops to trigger stop loss")
        print("-" * 40)
        logger.info("\n----- Price drops to trigger stop loss -----")
        stop_loss_price = decision['stop_loss']
        current_price = binance_client.mock_price
        percent_to_sl = (stop_loss_price / current_price) - 1
        
        await asyncio.sleep(1)
        tps, sls = binance_client.simulate_price_move(symbol, percent_to_sl)
        await strategy_manager.check_trade_completion(symbol)
        
        if sls:
            logger.info(f"Stop loss triggered! Orders filled: {len(sls)}")
        
        # Check the state of the trade after stop loss
        if strategy_manager.current_trade is None:
            logger.info("Trade was closed due to stop loss!")
        else:
            logger.info("Trade is still active.")
        
    else:
        logger.info(f"Trade not executed. Reason: {result['reason']}")
    
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETED")
    print("=" * 50)
    logger.info("Simulation completed!")
    return True


async def main():
    """Main function to run the trading simulation."""
    try:
        await simulate_trading()
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main()) 