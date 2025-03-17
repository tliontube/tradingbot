import logging
import json

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.trade_history = []
        self.trade_history_file = 'trade_history.json'

    async def process_trading_decision(self, symbol, gemini_decision, current_trade, trade_executor):
        """Process the trading decision from Gemini AI."""
        if current_trade is not None:
            return {"decision": "waiting", "reason": "Waiting for current trade to complete"}

        # Validate the decision
        entry_price = gemini_decision["entry_price"]
        stop_loss = gemini_decision["stop_loss"]
        take_profit = gemini_decision["take_profit"]

        if gemini_decision["decision"] == "buy":
            if stop_loss >= entry_price or take_profit <= entry_price:
                return {"decision": "waiting", "reason": "Invalid stop loss or take profit levels"}

        elif gemini_decision["decision"] == "sell":
            if stop_loss <= entry_price or take_profit >= entry_price:
                return {"decision": "waiting", "reason": "Invalid stop loss or take profit levels"}

        # Execute the trade
        trade_result = await trade_executor.execute_trade(symbol, gemini_decision)
        if not trade_result:
            return {"decision": "waiting", "reason": "Trade execution failed"}

        return gemini_decision

    def analyze_trade(self, trade_data):
        """Analyze a completed trade and extract patterns."""
        try:
            if 'pnl_percent' not in trade_data:
                logger.info("Trade data missing 'pnl_percent'. Estimating value.")
                trade_data['pnl_percent'] = self._estimate_pnl_percent(trade_data)

            # Add trade to history
            self.trade_history.append(trade_data)

            # Save updated trade history
            self._save_trade_history()

        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")

    def _save_trade_history(self):
        """Save trade history to a file."""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info(f"Trade history saved to {self.trade_history_file}")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def _estimate_pnl_percent(self, trade_data):
        """Estimate pnl_percent based on entry and exit prices."""
        entry_price = trade_data.get('entry_price', 1)
        exit_price = trade_data.get('exit_price', 1)
        return ((exit_price - entry_price) / entry_price) * 100
