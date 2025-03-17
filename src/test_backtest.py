import asyncio
from datetime import datetime, timedelta
import logging
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot
from backtesting import Backtester
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")

async def run_backtest():
    try:
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize clients with backtest mode
        binance_client = BinanceClient(
            config["binance"]["api_key"],
            config["binance"]["api_secret"],
            demo_mode=True  # Always use testnet for backtesting
        )
        
        # Initialize other components
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        telegram_bot = TelegramBot(config["telegram"]["bot_token"], config["telegram"]["chat_id"])
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)

        # Initialize futures settings
        symbols = ["ETHUSDT", "XRPUSDT", "TRXUSDT"]  # Removed BTCUSDT
        for symbol in symbols:
            await binance_client.initialize_futures_settings(symbol, config)

        # Set backtest period - reduce to 7 days to avoid rate limits
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Changed from 30 to 7 days
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Testing symbols: {symbols}")

        # Initialize backtester with rate limiting
        backtester = Backtester(strategy_manager, start_date, end_date)

        # Run backtest
        await backtester.run_backtest(symbols)

        # Clean up
        await binance_client.close_connection()
        await telegram_bot.stop()
        
        logger.info("Backtest completed successfully")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Ensure connections are closed
        try:
            await binance_client.close_connection()
            await telegram_bot.stop()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(run_backtest())
