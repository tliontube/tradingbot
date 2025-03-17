from datetime import datetime

class PerformanceTracker:
    def __init__(self, binance_client, telegram_bot):
        self.binance_client = binance_client
        self.telegram_bot = telegram_bot
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.initial_balance = 0
        self.current_balance = 0
        self.total_profit_loss = 0
        self.trade_history = []

    async def initialize_balance_tracking(self):
        """Initialize balance tracking by getting current account balance."""
        balances = await self.binance_client.get_account_balance()
        if balances and "USDT" in balances:
            self.initial_balance = balances["USDT"]["total"]
            self.current_balance = self.initial_balance
            await self.telegram_bot.send_message(
                f"🏦 *Initial Account Balance*: {self.initial_balance:.2f} USDT"
            )

    def calculate_performance_metrics(self):
        """Calculate current performance metrics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        profit_percentage = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_profit_loss": self.total_profit_loss,
            "profit_percentage": profit_percentage
        }

    async def update_performance_metrics(self, trade_result, profit_loss):
        """Update performance metrics after a trade."""
        self.total_trades += 1
        if trade_result == "Win":
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.total_profit_loss += profit_loss
        self.current_balance += profit_loss

        # Add trade to history
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": trade_result,
            "profit_loss": profit_loss,
            "balance": self.current_balance
        })
