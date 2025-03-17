import asyncio
import yaml
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

async def main():
    try:
        # Create required directories
        Path("/home/thomas/tradingbot/data/models").mkdir(parents=True, exist_ok=True)
        Path("/home/thomas/tradingbot/backtest_reports").mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Get trading pairs from config
        trading_pairs_config = config.get("binance", {}).get("trading_pairs", [])
        trading_pairs = [pair["symbol"] for pair in trading_pairs_config if pair.get("enabled", True)]
        
        # If no pairs are configured or enabled, use default pairs
        if not trading_pairs:
            trading_pairs = ["ETHUSDT", "XRPUSDT", "TRXUSDT"]  # Removed BTCUSDT
            print("No trading pairs configured or enabled, using defaults:", trading_pairs)
        else:
            print("Trading pairs loaded from config:", trading_pairs)

        # Initialize clients
        binance_client = BinanceClient(config["binance"]["api_key"], config["binance"]["api_secret"], config["binance"]["demo_mode"])
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        telegram_bot = TelegramBot(config["telegram"]["bot_token"], config["telegram"]["chat_id"])
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)

        # Initialize all components
        await strategy_manager.initialize()
        logger.info("Strategy Manager initialized successfully.")

        # Start the Telegram bot
        await telegram_bot.start()
        logger.info("Telegram bot started.")
        
        # Initialize futures settings if using futures
        if (binance_client.using_futures):
            # Initialize futures settings for all trading pairs
            for pair in trading_pairs:
                await binance_client.initialize_futures_settings(pair, config)
            
            # Check testnet account balance
            account = await binance_client.client.futures_account()
            assets = account.get('assets', [])
            usdt_asset = next((asset for asset in assets if asset['asset'] == 'USDT'), None)
            
            if usdt_asset:
                available_balance = float(usdt_asset.get('availableBalance', 0))
                wallet_balance = float(usdt_asset.get('walletBalance', 0))
                
                # If balance is still low after trying to fund
                if wallet_balance < 50:
                    await telegram_bot.send_message(
                        "⚠️ *Low Balance Warning* ⚠️\n\n"
                        f"Your testnet account has only {wallet_balance:.2f} USDT.\n"
                        "Trading may fail due to insufficient funds.\n\n"
                        "To get testnet funds:\n"
                        "1. Log in to https://testnet.binancefuture.com\n"
                        "2. Click on 'Get Assets'\n"
                        "3. Request test assets\n\n"
                        "Trading will continue with minimum quantities."
                    )
        
        # Initialize balance tracking
        await strategy_manager.initialize_balance_tracking()

        # Send initial status message
        await telegram_bot.send_message(
            f"🤖 *Trading Bot Started* 🤖\n\n"
            f"📈 *Gemini AI is in learning mode*. It will take more trades to gather data for ML training.\n"
            f"📊 *Trading Mode*: {'Futures' if binance_client.using_futures else 'Spot'}\n"
            f"💪 *Leverage*: {binance_client.futures_leverage}x\n"
            f"📈 *Monitored Pairs*: {', '.join(trading_pairs)}\n"
            f"⚙️ *Timeframe*: 1 minute\n"
            f"🎯 *Risk/Reward*: Minimum 1:1\n"
            f"🧠 *Strategy*: Basic price action with dynamic ML insights\n"
            f"🔄 *Auto-compound*: Enabled\n"
            f"👀 *Gemini will improve over time with ML training...*"
        )

        # Main loop
        while True:
            try:
                # Check if the current trade is complete
                if await strategy_manager.check_trade_completion():
                    # Analyze all markets and find the best trading opportunity
                    best_opportunity = await strategy_manager.find_best_opportunity(trading_pairs)
                    print("Best opportunity:", best_opportunity)

                    # No opportunities found
                    if best_opportunity is None or best_opportunity["decision_data"]["decision"] == "waiting":
                        await asyncio.sleep(30)  # Reduced waiting time from 60 to 30 seconds
                        continue
                        
                    # Execute trade for the best opportunity
                    await strategy_manager.execute_best_opportunity(best_opportunity)
                    
                elif strategy_manager.current_trade is not None:
                    # If a trade is active, print current status
                    symbol = strategy_manager.current_trade.get("symbol", "Unknown")
                    print(f"Active trade on {symbol}, waiting for completion")

                await asyncio.sleep(15)  # Reduced from 20 seconds
            except Exception as e:
                error_message = f"⚠️ *Error in Main Loop*: {e}"
                print(error_message)
                await telegram_bot.send_message(error_message)
                await asyncio.sleep(30)  # Reduced retry wait time
    except Exception as e:
        logger.error(f"❌ *Initialization Error*: {e}")
        await telegram_bot.send_message(f"❌ *Initialization Error*: {e}")
    finally:
        await telegram_bot.stop()
        await binance_client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")