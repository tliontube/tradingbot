import asyncio
import yaml
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot

async def main():
    try:
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Get trading pairs from config
        trading_pairs_config = config.get("binance", {}).get("trading_pairs", [])
        trading_pairs = [pair["symbol"] for pair in trading_pairs_config if pair.get("enabled", True)]
        
        # If no pairs are configured or enabled, use default pairs
        if not trading_pairs:
            trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
            print("No trading pairs configured or enabled, using defaults:", trading_pairs)
        else:
            print("Trading pairs loaded from config:", trading_pairs)

        # Initialize clients
        binance_client = BinanceClient(config["binance"]["api_key"], config["binance"]["api_secret"], config["binance"]["demo_mode"])
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        telegram_bot = TelegramBot(config["telegram"]["bot_token"], config["telegram"]["chat_id"])
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)

        # Start the Telegram bot
        await telegram_bot.start()
        
        # Initialize futures settings if using futures
        if binance_client.using_futures:
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
            "🤖 *Trading Bot Started* 🤖\n\n"
            "📈 Trading with Multi-Pair SMC Strategy\n"
            f"📊 Trading Mode: {'Futures' if binance_client.using_futures else 'Spot'}\n"
            f"💪 Leverage: {binance_client.futures_leverage}x\n"
            f"📈 Monitored Pairs: {', '.join(trading_pairs)}\n"
            "⚙️ Timeframe: 1 minute\n"
            "🎯 Risk/Reward: 1:2\n"
            "🔄 Auto-compound: Enabled\n"
            f"💸 Trading Fee: {config['binance'].get('fee_rate', 0.0004)*100:.3f}%\n\n"
            "Scanning multiple pairs for the best trading opportunities... 👀"
        )

        # Main loop
        while True:
            try:
                # Check if the current trade is complete
                if await strategy_manager.check_trade_completion():
                    # Analyze all markets and find the best trading opportunity
                    best_opportunity = await strategy_manager.find_best_opportunity(trading_pairs)
                    print("Best opportunity:", best_opportunity)  # Print the best opportunity for debugging

                    # No opportunities found
                    if best_opportunity is None or best_opportunity["decision"] == "waiting":
                        await asyncio.sleep(30)  # Wait before checking again
                        continue
                        
                    # Execute trade for the best opportunity
                    await strategy_manager.execute_best_opportunity(best_opportunity)
                    
                    # Send message to Telegram about the best opportunity
                    decision = best_opportunity["decision_data"]
                    symbol = best_opportunity["symbol"]
                    
                    # Ensure decision has all required fields for messaging
                    if decision is None:
                        continue
                        
                    # Set default values for missing fields
                    decision_trend = decision.get("trend", "unknown")
                    decision_strength = decision.get("trend_strength", "unknown")
                    
                    # Ensure structure_analysis exists and has default values
                    if "structure_analysis" not in decision:
                        decision["structure_analysis"] = {
                            "current_structure": "ranging",
                            "last_bos_level": 0,
                            "key_order_blocks": [],
                            "liquidity_pools": []
                        }
                    
                    # Send message to Telegram for the best opportunity
                    if decision["decision"] in ["buy", "sell"]:
                        message = (
                            f"🚨 *New Trading Signal* 🚨\n\n"
                            f"📊 **Symbol**: {symbol}\n"
                            f"⭐ **Opportunity Score**: {best_opportunity['score']:.2f}\n"
                            f"🔄 **Action**: {decision['decision'].upper()}\n"
                            f"📝 **Reason**: {decision['reason']}\n"
                            f"💰 **Entry Price**: {decision['entry_price']}\n"
                            f"🛑 **Stop-Loss**: {decision['stop_loss']}\n"
                            f"🎯 **Take-Profit**: {decision['take_profit']}\n"
                            f"📈 **Trend**: {decision_trend.capitalize()}\n"
                            f"💪 **Trend Strength**: {decision_strength.capitalize()}\n"
                            f"🏗️ **Structure Analysis**:\n"
                            f"   • Current Structure: {decision['structure_analysis']['current_structure'].capitalize()}\n"
                            f"   • Last BOS Level: {decision['structure_analysis']['last_bos_level']}\n"
                            f"   • Key Order Blocks: {decision['structure_analysis']['key_order_blocks']}\n"
                            f"   • Liquidity Pools: {decision['structure_analysis']['liquidity_pools']}\n\n"
                            f"Trading the BEST opportunity right now! 🚀"
                        )
                        await telegram_bot.send_message(message)
                    
                elif strategy_manager.current_trade is not None:
                    # If a trade is active, print current status
                    symbol = strategy_manager.current_trade.get("symbol", "Unknown")
                    print(f"Active trade on {symbol}, waiting for completion")

                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                print(traceback.format_exc())
                await asyncio.sleep(60)  # Wait before retrying
    except Exception as e:
        print(f"Initialization error: {e}")
    finally:
        await telegram_bot.stop()
        await binance_client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")