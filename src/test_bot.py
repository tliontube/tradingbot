import asyncio
import logging
import yaml
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_bot")

async def test_bot():
    """Test the bot's ability to execute trades, set stop loss, and take profit for multiple pairs."""
    # Load configuration from config.yaml
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize clients
    binance_client = BinanceClient(
        config["binance"]["api_key"],
        config["binance"]["api_secret"],
        config["binance"]["demo_mode"],
    )
    gemini_client = GeminiClient(config["gemini"]["api_key"])
    telegram_bot = TelegramBot(
        config["telegram"]["bot_token"], config["telegram"]["chat_id"]
    )
    strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)

    # Start Telegram bot
    await telegram_bot.start()

    # Test working symbols excluding BTCUSDT
    symbols = ["ETHUSDT", "XRPUSDT", "TRXUSDT"]  # Removed BTCUSDT

    try:
        for symbol in symbols:
            logger.info(f"Testing for symbol: {symbol}")

            # Step 1: Fetch market data
            logger.info("Fetching market data...")
            market_data = await binance_client.get_market_data(symbol, interval="1m", limit=200)
            if not market_data:
                logger.error(f"Failed to fetch market data for {symbol}.")
                continue

            # Step 2: Mock a trading decision
            logger.info("Mocking a trading decision...")
            decision = {
                "decision": "buy",
                "entry_price": market_data["closing_prices"][-1],
                "stop_loss": market_data["closing_prices"][-1] * 0.99,  # 1% below entry
                "take_profit": market_data["closing_prices"][-1] * 1.02,  # 2% above entry
                "reason": "Mock decision for testing",
            }

            # Step 3: Execute the trade
            logger.info("Executing trade...")
            trade_result = await strategy_manager.process_trading_decision(symbol, decision)
            if trade_result["decision"] == "waiting":
                logger.error(f"Trade not executed for {symbol}: {trade_result['reason']}")
                continue

            logger.info(f"Trade executed successfully for {symbol}.")

            # Step 4: Wait for 10 seconds before closing all orders
            logger.info(f"Waiting for 10 seconds before closing all orders for {symbol}...")
            await asyncio.sleep(10)

            # Step 5: Close all orders for the current symbol
            logger.info(f"Closing all orders for {symbol}...")
            await binance_client.cancel_all_orders(symbol)

            # Close the open position for the symbol
            logger.info(f"Closing the open position for {symbol}...")
            position_closed = await strategy_manager.trade_executor.close_position(symbol)
            if not position_closed:
                logger.error(f"Failed to close the position for {symbol}.")
                continue

            # Reset current trade in StrategyManager
            strategy_manager.current_trade = None

            # Step 6: Proceed to the next symbol
            logger.info(f"Finished testing for {symbol}. Moving to the next pair.")

    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        # Stop Telegram bot and close Binance client
        await telegram_bot.stop()
        await binance_client.close_connection()

if __name__ == "__main__":
    asyncio.run(test_bot())
