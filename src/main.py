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

        # Initialize clients
        binance_client = BinanceClient(config["binance"]["api_key"], config["binance"]["api_secret"], config["binance"]["demo_mode"])
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        telegram_bot = TelegramBot(config["telegram"]["bot_token"], config["telegram"]["chat_id"])
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot)

        # Start the Telegram bot
        await telegram_bot.start()

        # Main loop
        while True:
            try:
                # Check if the current trade is complete
                if await strategy_manager.check_trade_completion("BTCUSDT"):
                    # Analyze the market and make a trading decision
                    decision = await strategy_manager.decide_strategy("BTCUSDT")
                    print("Decision:", decision)  # Print the full decision for debugging

                    # If the decision is not "hold", send a message to Telegram
                    if decision["decision"] != "hold":
                        message = (
                            f"🚨 *New Trading Signal* 🚨\n\n"
                            f"📊 **Decision**: {decision['decision'].upper()}\n"
                            f"📝 **Reason**: {decision['reason']}\n"
                            f"💰 **Entry Price**: {decision['entry_price']}\n"
                            f"🛑 **Stop-Loss**: {decision['stop_loss']}\n"
                            f"🎯 **Take-Profit**: {decision['take_profit']}\n"
                            f"📈 **Trend**: {decision['trend'].capitalize()}\n"
                            f"💪 **Trend Strength**: {decision['trend_strength'].capitalize()}\n\n"
                            f"Good luck! 🚀"
                        )
                        await telegram_bot.send_message(message)

                await asyncio.sleep(300)  # Wait 1 minute before the next iteration
            except Exception as e:
                print(f"Error in main loop: {e}")
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