#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import yaml
import json
from datetime import datetime
import time

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
logger = logging.getLogger("live_test")

async def load_config():
    """Load configuration file."""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

async def run_live_test():
    """Test the trading bot in a live environment using testnet APIs."""
    binance_client = None
    try:
        print("=" * 80)
        print("STARTING LIVE TESTNET TRADING TEST")
        print("=" * 80)
        logger.info("Starting live testnet trading test...")
        
        # Load configuration
        config = await load_config()
        
        # Ensure we're using testnet for safety
        if not config["binance"].get("testnet", False):
            logger.warning("WARNING: Setting testnet=True for safety")
            config["binance"]["testnet"] = True
        
        # Create real clients, but connected to testnet
        binance_client = BinanceClient(
            config["binance"]["api_key"],
            config["binance"]["api_secret"],
            demo_mode=True  # Force demo mode (testnet) for safety
        )
        
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        
        telegram_bot = TelegramBot(
            config["telegram"].get("API_TOKEN"),
            config["telegram"].get("CHAT_ID")
        )
        
        # Create strategy manager
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)
        
        # Optional: Adjust minimum hold time for faster testing
        if input("Set minimum hold time to 0 seconds for testing? (y/n): ").lower() == 'y':
            strategy_manager.min_hold_time_seconds = 0
            logger.info(f"Setting min_hold_time_seconds to 0 for testing")
        
        # Set the symbol to test
        symbol_config = config["binance"].get("trading_pairs", ["BTCUSDT"])[0]
        if isinstance(symbol_config, dict):
            symbol = symbol_config.get("symbol", "BTCUSDT")
        else:
            symbol = symbol_config
        logger.info(f"Testing with symbol: {symbol}")
        
        # Step 1: Initialize clients and verify connectivity
        print("\n" + "=" * 50)
        print("STEP 1: INITIALIZING CLIENTS")
        print("=" * 50)
        logger.info("Initializing and verifying client connectivity...")
        
        # Test Binance connection
        try:
            account_info = await binance_client.get_account_balance()
            logger.info(f"Binance connection successful. Available assets: {list(account_info.keys())}")
        except Exception as e:
            logger.error(f"Failed to connect to Binance testnet: {e}")
            return False
        
        # Step 2: Get market data
        print("\n" + "=" * 50)
        print("STEP 2: FETCHING MARKET DATA")
        print("=" * 50)
        logger.info("Fetching market data...")
        
        start_time = time.time()
        market_data = await binance_client.get_market_data(symbol, interval="5m", limit=500)
        end_time = time.time()
        
        logger.info(f"Market data fetched in {end_time - start_time:.2f} seconds")
        logger.info(f"Market data structure: {type(market_data)}")
        if market_data:
            logger.info(f"Market data keys: {market_data.keys() if hasattr(market_data, 'keys') else 'No keys'}")
            
        # Adapt market data to expected format
        adapted_market_data = {}
        if isinstance(market_data, dict):
            # Map the keys from the API response to the expected format
            key_mapping = {
                'closing_prices': 'close',
                'volumes': 'volume',
                'high_prices': 'high',
                'low_prices': 'low',
                'open_prices': 'open'
            }
            
            for api_key, expected_key in key_mapping.items():
                if api_key in market_data:
                    adapted_market_data[expected_key] = market_data[api_key]
        
        if 'close' in adapted_market_data:
            current_price = adapted_market_data['close'][-1]
            logger.info(f"Current price for {symbol}: {current_price}")
        else:
            # Try to get the latest price directly
            current_price = await binance_client.get_latest_price(symbol)
            logger.info(f"Current price (from get_latest_price) for {symbol}: {current_price}")
        
        # Ask if user wants to force a trade for testing
        force_trade = input("Do you want to force a trade to test take-profit levels? (y/n): ").lower() == 'y'
        
        if force_trade:
            print("\n" + "=" * 50)
            print("STEP 3: CREATING FORCED TEST TRADE")
            print("=" * 50)
            logger.info("Creating forced test trade with multiple take-profit levels...")
            
            # Ask for trade direction
            trade_direction = input("Enter trade direction (buy/sell): ").lower()
            if trade_direction not in ["buy", "sell"]:
                logger.error(f"Invalid trade direction: {trade_direction}. Must be 'buy' or 'sell'.")
                return False
            
            # Get current price if not already fetched
            if not current_price:
                current_price = await binance_client.get_latest_price(symbol)
            
            # Calculate stop loss and take profit levels based on direction
            if trade_direction == "buy":
                # For buy: Stop loss 3% below entry, take profits at 3%, 6%, and 9% above entry
                stop_loss = current_price * 0.97
                take_profit_levels = {
                    "tp1": current_price * 1.03,
                    "tp2": current_price * 1.06, 
                    "tp3": current_price * 1.09
                }
            else:  # sell
                # For sell: Stop loss 3% above entry, take profits at 3%, 6%, and 9% below entry
                stop_loss = current_price * 1.03
                take_profit_levels = {
                    "tp1": current_price * 0.97,
                    "tp2": current_price * 0.94,
                    "tp3": current_price * 0.91
                }
            
            # Create a mock decision
            decision = {
                "decision": trade_direction,
                "reason": "Forced test trade to verify take-profit functionality",
                "confidence": 10,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit_levels["tp2"],  # Middle target as primary take profit
                "take_profit_levels": take_profit_levels,
                "trend": "uptrend" if trade_direction == "buy" else "downtrend",
                "trend_strength": "strong",
                "risk_reward_ratio": 2.0,
                "structure_analysis": {
                    "current_structure": "bullish" if trade_direction == "buy" else "bearish",
                    "last_bos_level": current_price * 0.95,
                    "key_order_blocks": [
                        {
                            "type": "bullish" if trade_direction == "buy" else "bearish",
                            "price": current_price * (0.96 if trade_direction == "buy" else 1.04),
                            "status": "active"
                        }
                    ],
                    "market_structure": {
                        "structure_type": "HH-HL" if trade_direction == "buy" else "LH-LL",
                    }
                }
            }
            
            logger.info(f"Created forced {trade_direction.upper()} decision at price {current_price}")
            logger.info(f"Stop loss: {stop_loss}")
            logger.info(f"Take profit levels: {take_profit_levels}")
            
        else:
            # Step 3: Get trading decision from Gemini
            print("\n" + "=" * 50)
            print("STEP 3: GETTING TRADING DECISION")
            print("=" * 50)
            logger.info("Getting trading decision from Gemini model...")
            try:
                decision = gemini_client.get_trading_decision(
                    adapted_market_data.get('close', []),
                    adapted_market_data.get('volume', []),
                    adapted_market_data.get('high', []),
                    adapted_market_data.get('low', []),
                    adapted_market_data.get('open', []),
                    symbol
                )
                
                logger.info(f"Decision: {decision['decision']} at {decision.get('entry_price')}")
                if decision.get('take_profit_levels'):
                    logger.info(f"Take profit levels: {decision['take_profit_levels']}")
                
            except Exception as e:
                logger.error(f"Failed to get trading decision: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        
        # Step 4: Confirm execution
        print("\n" + "=" * 50)
        print("STEP 4: CONFIRM EXECUTION")
        print("=" * 50)
        
        if decision["decision"] == "waiting" and not force_trade:
            logger.info(f"No trade to execute. Reason: {decision.get('reason', 'No reason provided')}")
            return True
        
        confirm = input(f"Execute {decision['decision']} order for {symbol} at {decision['entry_price']}? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Trade execution canceled by user")
            return True
        
        # Step 5: Process and execute the trading decision
        print("\n" + "=" * 50)
        print("STEP 5: PROCESSING TRADING DECISION")
        print("=" * 50)
        logger.info("Processing trading decision...")
        try:
            result = await strategy_manager.process_trading_decision(symbol, decision)
            
            if result.get("success", False):
                logger.info("Trade executed successfully!")
                
                # Get open orders
                open_orders = await binance_client.get_open_orders(symbol)
                logger.info(f"Open orders: {len(open_orders)}")
                
                for order in open_orders:
                    order_type = order.get("type", "UNKNOWN")
                    if "STOP" in order_type:
                        logger.info(f"Stop loss order: {order}")
                    elif "TAKE_PROFIT" in order_type:
                        logger.info(f"Take profit order: {order}")
                
                # Monitor the trade
                monitor = input("Monitor the trade for price movements? (y/n): ")
                if monitor.lower() == 'y':
                    monitoring_duration = int(input("Enter monitoring duration in seconds (max 300): "))
                    monitoring_duration = min(monitoring_duration, 300)  # Cap at 5 minutes
                    
                    logger.info(f"Monitoring trade for {monitoring_duration} seconds...")
                    monitoring_interval = 5  # Check every 5 seconds
                    
                    start_time = datetime.now()
                    while (datetime.now() - start_time).total_seconds() < monitoring_duration:
                        # Get current price
                        current_price = await binance_client.get_latest_price(symbol)
                        logger.info(f"Current price: {current_price}")
                        
                        # Check position status
                        position = await binance_client.get_position_info(symbol)
                        if position:
                            logger.info(f"Position: {position}")
                            
                            # Calculate unrealized PnL
                            entry_price = float(position.get("entryPrice", 0))
                            position_amt = float(position.get("positionAmt", 0))
                            
                            if entry_price > 0 and position_amt != 0:
                                if position_amt > 0:  # Long position
                                    pnl_percent = (current_price / entry_price - 1) * 100
                                else:  # Short position
                                    pnl_percent = (entry_price / current_price - 1) * 100
                                    
                                logger.info(f"Unrealized PnL: {pnl_percent:.2f}%")
                        
                        # Check if trade has been completed
                        await strategy_manager.check_trade_completion(symbol)
                        
                        if strategy_manager.current_trade is None:
                            logger.info("Trade has been completed or closed")
                            break
                        
                        await asyncio.sleep(monitoring_interval)
                    
                    logger.info("Monitoring completed")
            else:
                logger.warning(f"Trade execution failed. Reason: {result.get('reason', 'Unknown reason')}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        # Clean up by canceling any remaining orders if requested
        if input("Cancel all remaining orders? (y/n): ").lower() == 'y':
            try:
                await binance_client.cancel_all_open_orders(symbol)
                logger.info("All open orders canceled")
            except Exception as e:
                logger.error(f"Error canceling orders: {e}")
        
        print("\n" + "=" * 50)
        print("LIVE TEST COMPLETED")
        print("=" * 50)
        logger.info("Live test completed!")
        return True
    except Exception as e:
        logger.error(f"Error in live test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Clean up resources
        if binance_client:
            logger.info("Closing Binance client connection...")
            await binance_client.close_connection()

async def main():
    """Main function to run the live test."""
    try:
        await run_live_test()
    except Exception as e:
        logger.error(f"Error in live test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 