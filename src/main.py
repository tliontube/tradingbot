import asyncio
import yaml
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot
import time

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
            trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
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
            "📈 Trading with Advanced SMC Strategy and Professional Decision System\n"
            f"📊 Trading Mode: {'Futures' if binance_client.using_futures else 'Spot'}\n"
            f"💪 Leverage: {binance_client.futures_leverage}x\n"
            f"📈 Monitored Pairs: {', '.join(trading_pairs)}\n"
            "⚙️ Timeframe: 5 minutes\n"
            "🎯 Risk/Reward: Minimum 1:2, preferably 1:3+\n"
            "🧠 Strategy: Smart Money Concepts with Order Blocks, Fair Value Gaps, BOS and CHoCH\n"
            "⏳ Trading Style: Balanced selectivity, trading high-quality setups while not missing opportunities\n"
            "🔬 Quality Score: Trades scoring 65+ on the opportunity matrix will be executed\n"
            "🔄 Auto-compound: Enabled\n"
            f"💸 Trading Fee: {config['binance'].get('fee_rate', 0.0004)*100:.3f}%\n\n"
            "Patiently scanning the market for perfect trading opportunities... 👀"
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
                    if best_opportunity is None or best_opportunity["decision_data"]["decision"] == "waiting":
                        # Send notification about waiting for better opportunities
                        if best_opportunity and "decision_data" in best_opportunity and "reason" in best_opportunity["decision_data"]:
                            reason = best_opportunity["decision_data"]["reason"]
                            # Only send message occasionally to avoid spam (every 10 minutes)
                            current_time = time.time()
                            if not hasattr(strategy_manager, "last_waiting_notification_time") or \
                               current_time - getattr(strategy_manager, "last_waiting_notification_time", 0) > 600:
                                await telegram_bot.send_message(
                                    f"⏳ *Smart Trading Strategy: Searching for Quality Setup* ⏳\n\n"
                                    f"The AI is analyzing market structure across all pairs and is looking for a high-quality entry opportunity.\n\n"
                                    f"*Current Analysis*:\n"
                                    f"• {reason}\n\n"
                                    f"*Our Balanced Approach*:\n"
                                    f"• Trading setup quality must score at least 65/100\n"
                                    f"• Only taking trades with positive risk-reward ratios\n"
                                    f"• Looking for setups with clear institutional signals\n"
                                    f"• Finding opportunities without being overly selective\n\n"
                                    f"The trading algorithm will continue monitoring for a combination of Order Blocks, Break of Structure (BOS), Change of Character (CHoCH), and Fair Value Gaps (FVG) that indicate high-probability entries."
                                )
                                # Update last notification time
                                strategy_manager.last_waiting_notification_time = current_time
                        
                        await asyncio.sleep(60)  # Increased from 30 to 60 seconds - wait more patiently before checking again
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
                        # Check if we have the new multi-level take profit format
                        take_profit_message = ""
                        if "take_profit_levels" in decision and isinstance(decision["take_profit_levels"], dict):
                            if "tp1" in decision["take_profit_levels"]:
                                take_profit_message = (
                                    f"🎯 **Take-Profit Levels**:\n"
                                    f"   • TP1 (1:1): {decision['take_profit_levels']['tp1']}\n"
                                    f"   • TP2 (1:2): {decision['take_profit_levels']['tp2']}\n"
                                    f"   • TP3 (1:3+): {decision['take_profit_levels']['tp3']}\n"
                                )
                        else:
                            take_profit_message = f"🎯 **Take-Profit**: {decision['take_profit']}\n"
                        
                        # Check for confidence score
                        confidence_message = ""
                        risk_reward_message = ""
                        
                        if "confidence" in decision:
                            confidence = decision["confidence"]
                            confidence_emoji = "🟢" if confidence >= 8 else "🟡" if confidence >= 6 else "🟠"
                            confidence_message = f"{confidence_emoji} **Confidence**: {confidence}/10\n"
                            
                        if "risk_reward_ratio" in decision:
                            risk_reward = decision["risk_reward_ratio"]
                            risk_reward_message = f"⚖️ **Risk/Reward**: 1:{risk_reward:.2f}\n"
                            
                        # Check for Wyckoff phase and volume analysis
                        wyckoff_message = ""
                        volume_message = ""
                        
                        if "structure_analysis" in decision:
                            structure = decision["structure_analysis"]
                            if "wyckoff_phase" in structure and structure["wyckoff_phase"] not in ["undefined", ""]:
                                wyckoff_message = f"📊 **Wyckoff Phase**: {structure['wyckoff_phase'].capitalize()}\n"
                                
                            if "volume_analysis" in structure and structure["volume_analysis"] not in ["undefined", "neutral", ""]:
                                volume_message = f"📈 **Volume Analysis**: {structure['volume_analysis'].capitalize()}\n"
                        
                        message = (
                            f"🚨 *New Trading Signal* 🚨\n\n"
                            f"📊 **Symbol**: {symbol}\n"
                            f"⭐ **Opportunity Score**: {best_opportunity['score']:.2f}\n"
                            f"🔄 **Action**: {decision['decision'].upper()}\n"
                            f"{confidence_message}"
                            f"{risk_reward_message}"
                            f"📝 **Reason**: {decision['reason']}\n"
                            f"💰 **Entry Price**: {decision['entry_price']}\n"
                            f"🛑 **Stop-Loss**: {decision['stop_loss']}\n"
                            f"{take_profit_message}"
                            f"📈 **Trend**: {decision_trend.capitalize()}\n"
                            f"💪 **Trend Strength**: {decision_strength.capitalize()}\n"
                            f"{wyckoff_message}"
                            f"{volume_message}"
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

                await asyncio.sleep(60)  # Increased from 30 to 60 seconds - check every minute instead of every 30 seconds
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