from datetime import datetime, timedelta
import asyncio
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy_manager, start_date, end_date):
        self.strategy_manager = strategy_manager
        self.start_date = start_date
        self.end_date = end_date
        self.results = []
        self.trade_history = []
        self.initial_balance = 1000
        self.current_balance = self.initial_balance
        self.telegram_bot = strategy_manager.telegram_bot
        self.last_api_call = 0
        self.api_call_interval = 1.5  # Minimum seconds between API calls

    async def run_backtest(self, symbols):
        """Run backtest on historical data."""
        symbols = ["ETHUSDT", "XRPUSDT", "TRXUSDT"]  # Removed BTCUSDT
        for symbol in symbols:
            historical_data = await self._fetch_historical_data(symbol)
            if not historical_data:
                continue

            print(f"\nBacktesting {symbol}...")
            
            for i in range(len(historical_data) - 200):
                window = historical_data[i:i+200]
                
                # Convert data for Gemini client
                window_data = {
                    'closing_prices': [float(k[4]) for k in window],
                    'volumes': [float(k[5]) for k in window],
                    'high_prices': [float(k[2]) for k in window],
                    'low_prices': [float(k[3]) for k in window],
                    'open_prices': [float(k[1]) for k in window]
                }
                
                # Rate limiting for Gemini API calls
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.api_call_interval:
                    await asyncio.sleep(self.api_call_interval - time_since_last_call)
                
                try:
                    decision = self.strategy_manager.gemini_client.get_trading_decision(
                        closing_prices=window_data['closing_prices'],
                        volumes=window_data['volumes'],
                        high_prices=window_data['high_prices'],
                        low_prices=window_data['low_prices'],
                        open_prices=window_data['open_prices'],
                        symbol=symbol
                    )
                    self.last_api_call = time.time()
                    
                    if decision['decision'] != 'waiting':
                        trade_result = self._simulate_trade(decision, historical_data[i+200:i+400])
                        if trade_result:
                            self.trade_history.append({
                                'symbol': symbol,
                                'entry_time': datetime.fromtimestamp(window[-1][0]/1000),
                                'entry_price': decision['entry_price'],
                                'direction': decision['decision'],
                                'exit_price': trade_result['exit_price'],
                                'exit_time': trade_result['exit_time'],
                                'pnl': trade_result['pnl'],
                                'pnl_percent': trade_result['pnl_percent']
                            })
                            self.current_balance *= (1 + trade_result['pnl_percent']/100)
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        logger.warning("Hit rate limit, pausing for 60 seconds...")
                        await asyncio.sleep(60)
                        continue
                    else:
                        logger.error(f"Error getting trading decision: {e}")
                        continue

        await self._generate_backtest_report()

    def _simulate_trade(self, decision, forward_data):
        """Simulate a trade execution and return the result."""
        entry_price = decision['entry_price']
        stop_loss = decision['stop_loss']
        take_profit = decision['take_profit']
        
        for candle in forward_data:
            timestamp = datetime.fromtimestamp(candle[0]/1000)
            high = float(candle[2])
            low = float(candle[3])
            
            if decision['decision'] == 'buy':
                if low <= stop_loss:
                    return {
                        'exit_price': stop_loss,
                        'exit_time': timestamp,
                        'pnl': stop_loss - entry_price,
                        'pnl_percent': ((stop_loss - entry_price) / entry_price) * 100
                    }
                if high >= take_profit:
                    return {
                        'exit_price': take_profit,
                        'exit_time': timestamp,
                        'pnl': take_profit - entry_price,
                        'pnl_percent': ((take_profit - entry_price) / entry_price) * 100
                    }
            else:  # sell
                if high >= stop_loss:
                    return {
                        'exit_price': stop_loss,
                        'exit_time': timestamp,
                        'pnl': entry_price - stop_loss,
                        'pnl_percent': ((entry_price - stop_loss) / entry_price) * 100
                    }
                if low <= take_profit:
                    return {
                        'exit_price': take_profit,
                        'exit_time': timestamp,
                        'pnl': entry_price - take_profit,
                        'pnl_percent': ((entry_price - take_profit) / entry_price) * 100
                    }
        return None

    async def _fetch_historical_data(self, symbol):
        """Fetch historical data from Binance."""
        try:
            klines = await self.strategy_manager.binance_client.client.futures_historical_klines(
                symbol=symbol,
                interval='1m',
                start_str=self.start_date.strftime('%Y-%m-%d'),
                end_str=self.end_date.strftime('%Y-%m-%d')
            )
            
            if not klines:
                logger.error(f"No historical data returned for {symbol}")
                return None

            return klines
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    async def _generate_backtest_report(self):
        """Generate comprehensive backtest report and send to Telegram."""
        if not self.trade_history:
            print("No trades executed during backtest period")
            return

        # Calculate metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = len([t for t in self.trade_history if t['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        wins = [t['pnl_percent'] for t in self.trade_history if t['pnl'] > 0]
        losses = [t['pnl_percent'] for t in self.trade_history if t['pnl'] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100

        # Generate report
        report_dir = Path('/home/thomas/tradingbot/backtest_reports')
        report_dir.mkdir(exist_ok=True)
        
        # Generate Telegram message
        telegram_message = (
            "🤖 *Backtest Results* 🤖\n\n"
            f"📅 *Period*: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"
            f"💰 *Initial Balance*: {self.initial_balance:.2f} USDT\n"
            f"💵 *Final Balance*: {self.current_balance:.2f} USDT\n"
            f"📈 *Total Return*: {total_return:.2f}%\n\n"
            f"📊 *Performance Metrics*:\n"
            f"• Total Trades: {total_trades}\n"
            f"• Winning Trades: {winning_trades}\n"
            f"• Losing Trades: {losing_trades}\n"
            f"• Win Rate: {win_rate:.2f}%\n"
            f"• Avg Win: {avg_win:.2f}%\n"
            f"• Avg Loss: {avg_loss:.2f}%\n"
            f"• Profit Factor: {profit_factor:.2f}\n\n"
            "🔍 *Best Trades*:\n"
        )

        # Add top 3 best trades
        best_trades = sorted(self.trade_history, key=lambda x: x['pnl_percent'], reverse=True)[:3]
        for trade in best_trades:
            telegram_message += (
                f"• {trade['symbol']}: {trade['pnl_percent']:.2f}% profit\n"
                f"  Entry: {trade['entry_price']:.2f} → Exit: {trade['exit_price']:.2f}\n"
            )

        # Save results to file and send to Telegram
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f'backtest_report_{report_time}.txt'
        
        with open(report_path, 'w') as f:
            f.write(telegram_message.replace('*', '').replace('•', '-'))
            
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        entry_times = [trade['entry_time'] for trade in self.trade_history]
        pnl_percents = [trade['pnl_percent'] for trade in self.trade_history]
        equity_curve = [self.initial_balance * (1 + sum(pnl_percents[:i+1])/100) for i in range(len(pnl_percents))]
        plt.plot(entry_times, equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Account Value (USDT)')
        plt.grid(True)
        equity_curve_path = report_dir / f'equity_curve_{report_time}.png'
        plt.savefig(equity_curve_path)
        plt.close()

        # Send report to Telegram
        asyncio.create_task(self.telegram_bot.send_message(telegram_message))
        
        # Send equity curve plot
        with open(equity_curve_path, 'rb') as f:
            asyncio.create_task(self.telegram_bot.send_photo(f))

        print(f"Backtest report saved to {report_path} and sent to Telegram")
