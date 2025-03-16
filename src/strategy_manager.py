class StrategyManager:
    def __init__(self, gemini_client, binance_client, telegram_bot):
        self.gemini_client = gemini_client
        self.binance_client = binance_client
        self.telegram_bot = telegram_bot
        self.current_trade = None  # Track the current trade

    async def decide_strategy(self, symbol):
        # Check if there is an active trade
        if self.current_trade is not None:
            return {"decision": "waiting", "reason": "Waiting for current trade to complete", "stop_loss": None, "take_profit": None}

        # Fetch market data
        market_data = await self.binance_client.get_market_data(symbol, interval="5m", limit=100)
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

        # If the decision is not "waiting", mark the current trade
        if gemini_decision["decision"] not in ["waiting"]:
            self.current_trade = {
                "symbol": symbol,
                "decision": gemini_decision["decision"],
                "stop_loss": gemini_decision["stop_loss"],
                "take_profit": gemini_decision["take_profit"],
                "entry_price": gemini_decision["entry_price"]
            }

        return gemini_decision

    async def check_trade_completion(self, symbol):
        # Check if the current trade is complete
        if self.current_trade is None:
            return True  # No active trade

        # Fetch the latest price
        latest_price = await self.binance_client.get_latest_price(symbol)
        if latest_price is None:
            return False  # Unable to check trade completion

        # Check if the price has hit stop-loss or take-profit
        stop_loss = self.current_trade["stop_loss"]
        take_profit = self.current_trade["take_profit"]
        entry_price = self.current_trade["entry_price"]

        if self.current_trade["decision"] == "buy":
            if latest_price <= stop_loss or latest_price >= take_profit:
                # Determine if the trade was a win or a loss
                if latest_price <= stop_loss:
                    result = "Loss"
                else:
                    result = "Win"
                # Send a Telegram message
                message = (
                    f"📢 *Trade Closed* 📢\n\n"
                    f"📊 **Symbol**: {symbol}\n"
                    f"📝 **Result**: {result}\n"
                    f"💰 **Entry Price**: {entry_price}\n"
                    f"🛑 **Stop-Loss**: {stop_loss}\n"
                    f"🎯 **Take-Profit**: {take_profit}\n"
                    f"📈 **Exit Price**: {latest_price}\n\n"
                    f"Better luck next time! 🚀" if result == "Loss" else "Great job! 🎉"
                )
                await self.telegram_bot.send_message(message)
                self.current_trade = None  # Trade completed
                return True
        elif self.current_trade["decision"] == "sell":
            if latest_price >= stop_loss or latest_price <= take_profit:
                # Determine if the trade was a win or a loss
                if latest_price >= stop_loss:
                    result = "Loss"
                else:
                    result = "Win"
                # Send a Telegram message
                message = (
                    f"📢 *Trade Closed* 📢\n\n"
                    f"📊 **Symbol**: {symbol}\n"
                    f"📝 **Result**: {result}\n"
                    f"💰 **Entry Price**: {entry_price}\n"
                    f"🛑 **Stop-Loss**: {stop_loss}\n"
                    f"🎯 **Take-Profit**: {take_profit}\n"
                    f"📈 **Exit Price**: {latest_price}\n\n"
                    f"Better luck next time! 🚀" if result == "Loss" else "Great job! 🎉"
                )
                await self.telegram_bot.send_message(message)
                self.current_trade = None  # Trade completed
                return True

        return False  # Trade is still active