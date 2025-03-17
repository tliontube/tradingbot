class NotificationManager:
    def __init__(self, telegram_bot):
        self.telegram_bot = telegram_bot

    async def notify_general(self, message):
        """Send a general notification."""
        await self.telegram_bot.send_message(message)

    async def notify_trade_opened(self, symbol, decision):
        """Send notification when a new trade is opened."""
        # Determine emojis based on trade direction
        direction_emoji = "🟢" if decision['decision'].lower() == "buy" else "🔴"
        
        message = (
            f"{direction_emoji} *New Trade Opened* {direction_emoji}\n\n"
            f"🎯 *Symbol*: {symbol}\n"
            f"📊 *Action*: {decision['decision'].upper()}\n"
            f"💰 *Entry Price*: {decision['entry_price']:.2f}\n"
            f"🛑 *Stop Loss*: {decision['stop_loss']:.2f}\n"
            f"✨ *Take Profit*: {decision['take_profit']:.2f}\n\n"
            f"⚡ *Reason*: {decision.get('reason', 'N/A')}\n\n"
            f"⚡ *Happy Trading!* 📈"
        )
        await self.telegram_bot.send_message(message)

    async def notify_trade_closed(self, symbol, result, entry_price, exit_price):
        """Send notification when a trade is closed."""
        try:
            # Calculate profit/loss percentage
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_emoji = "🎯" if result == "Win" else "📉"
            result_emoji = "✅" if result == "Win" else "❌"
            
            message = (
                f"{pnl_emoji} *Trade Closed* {pnl_emoji}\n\n"
                f"🎯 *Symbol*: {symbol}\n"
                f"{result_emoji} *Result*: {result}\n"
                f"📥 *Entry Price*: {entry_price:.2f}\n"
                f"📤 *Exit Price*: {exit_price:.2f}\n"
                f"📊 *PnL*: {pnl_pct:.2f}%\n\n"
                f"💫 *Keep Trading!* 💪"
            )
            self.telegram_bot.logger.info(f"Sending trade closed notification for {symbol}: {message}")
            await self.telegram_bot.send_message(message)
        except Exception as e:
            self.telegram_bot.logger.error(f"Error sending trade closed notification for {symbol}: {e}")

    async def notify_gemini_progress(self, message):
        """Send a notification about Gemini's progress."""
        await self.telegram_bot.send_message(f"🚀 *Gemini Progress*: {message}")
