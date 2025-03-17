import logging
from performance_tracker import PerformanceTracker
from trade_execution import TradeExecutor
from trade_analysis import TradeAnalyzer
from notifications import NotificationManager
from machine_learning.models import TradingModel
from pathlib import Path

class StrategyManager:
    def __init__(self, gemini_client, binance_client, telegram_bot, config=None):
        # Initialize logger first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.gemini_client = gemini_client
        self.binance_client = binance_client
        self.telegram_bot = telegram_bot
        self.config = config

        # Initialize components
        self.performance_tracker = PerformanceTracker(binance_client, telegram_bot)
        self.trade_executor = TradeExecutor(binance_client, config)
        self.trade_analyzer = TradeAnalyzer(config)
        self.notification_manager = NotificationManager(telegram_bot)
        self.trading_model = TradingModel()

        self.current_trade = None  # Track the current trade

        # Log the initialized parameters
        self.logger.info(f"StrategyManager initialized with config: {config}")

    async def initialize(self):
        """Initialize all components."""
        try:
            # Initialize ML models first
            await self._initialize_ml_models()
            # Initialize balance tracking
            await self.initialize_balance_tracking()
            # Initialize futures settings for all trading pairs
            trading_pairs = ["ETHUSDT", "XRPUSDT", "TRXUSDT"]  # Removed BTCUSDT
            for pair in trading_pairs:
                await self.binance_client.initialize_futures_settings(pair, self.config)
            # Notify user about initialization
            await self.telegram_bot.send_message("✅ *Strategy Manager Initialized*")
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            await self.telegram_bot.send_message(f"❌ *Initialization Error*: {e}")

    async def _initialize_ml_models(self):
        """Initialize ML models with proper training data."""
        initial_data = [
            # Positive examples
            {
                'pnl_percent': 2.0,
                'hold_time': 300,
                'volume_profile': {'relative_volume': 1.2},
                'market_structure': {'trend_strength': 0.8},
                'pre_entry_candles': [],
                'market_conditions': {'liquidity_sweep': 1},
                'entry_quality': 0.8
            },
            # Negative examples
            {
                'pnl_percent': -1.0,
                'hold_time': 150,
                'volume_profile': {'relative_volume': 0.8},
                'market_structure': {'trend_strength': 0.3},
                'pre_entry_candles': [],
                'market_conditions': {'liquidity_sweep': 0},
                'entry_quality': 0.3
            }
        ]
        
        # Log initial data balance
        pattern_labels = [1 if t['pnl_percent'] > 0 else 0 for t in initial_data]
        class_counts = {label: pattern_labels.count(label) for label in set(pattern_labels)}
        self.logger.info(f"Initial data class distribution: {class_counts}")
        
        # Train models
        await self.load_ml_models()
        self.trading_model.train(initial_data)

    async def load_ml_models(self):
        """Load or train ML models."""
        try:
            if not self.trading_model.load_models():
                self.logger.info("Training new models with initial data...")
                # Create initial training data with basic structure
                initial_data = [{
                    'pnl_percent': 0,
                    'hold_time': 0,
                    'volume_profile': {'relative_volume': 1},
                    'market_structure': {'trend_strength': 0},
                    'pre_entry_candles': [],
                    'market_conditions': {
                        'liquidity_sweep': 0,
                        'order_block_strength': 0,
                        'fvg_presence': 0,
                        'bos_confirmed': 0,
                        'choch_confirmed': 0
                    },
                    'entry_quality': 0.5
                }]
                self.trading_model.train(initial_data)
                self.logger.info("Initial model training complete")
        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}")

    async def initialize_balance_tracking(self):
        """Initialize balance tracking."""
        await self.performance_tracker.initialize_balance_tracking()

    async def process_trading_decision(self, symbol, gemini_decision):
        """Process trading decision with ML validation."""
        try:
            # Get current market data
            market_data = await self.binance_client.get_market_data(symbol)

            # Prepare data for ML models
            ml_input = {
                'pnl_percent': None,  # To be calculated after trade completion
                'hold_time': None,    # To be calculated after trade completion
                'volume_profile': {
                    'relative_volume': self._calculate_relative_volume(market_data)
                },
                'market_structure': {
                    'trend_strength': gemini_decision.get('trend_strength', 0)
                },
                'pre_entry_candles': market_data['closing_prices'][-10:],  # Last 10 candles
                'market_conditions': {
                    'liquidity_sweep': self._detect_liquidity_sweep(market_data),
                    'order_block_strength': self._calculate_order_block_strength(market_data),
                    # Add other conditions as needed
                }
            }

            # Get ML predictions
            ml_predictions = self.trading_model.predict(ml_input)

            # Validate ML predictions
            if ml_predictions['combined_score'] < 0.4:  # Adjust threshold as needed
                return {
                    'decision': 'waiting',
                    'reason': f"ML models insufficient confidence: {ml_predictions['combined_score']:.2f}"
                }

            # Adjust Gemini's decision dynamically
            gemini_decision['prompt'] = self._generate_dynamic_prompt(ml_predictions)

            return await self.trade_analyzer.process_trading_decision(
                symbol, gemini_decision, self.current_trade, self.trade_executor
            )

        except Exception as e:
            self.logger.error(f"Error processing trading decision: {e}")
            return {'decision': 'waiting', 'reason': str(e)}

    def _calculate_relative_volume(self, market_data):
        """Calculate relative volume from market data."""
        volumes = market_data['volumes']
        avg_volume = sum(volumes) / len(volumes)
        return volumes[-1] / avg_volume if avg_volume > 0 else 1

    def _detect_liquidity_sweep(self, market_data):
        """Detect liquidity sweeps from market data."""
        # Implement logic to detect liquidity sweeps
        return 0  # Placeholder

    def _calculate_order_block_strength(self, market_data):
        """Calculate order block strength from market data."""
        # Implement logic to calculate order block strength
        return 0  # Placeholder

    def _generate_dynamic_prompt(self, ml_predictions):
        """Generate a dynamic prompt for Gemini based on ML insights."""
        return f"""
        ## DYNAMIC PROMPT BASED ON ML INSIGHTS
        
        • Pattern Probability: {ml_predictions['pattern_probability']:.2f}
        • Entry Quality: {ml_predictions['entry_quality']:.2f}
        • Risk Score: {ml_predictions['risk_score']:.2f}
        
        Use these insights to refine your trading decisions.
        """

    def _adjust_trade_parameters(self, gemini_decision, ml_predictions):
        """Adjust trade parameters based on ML predictions."""
        try:
            # Copy original decision
            adjusted = gemini_decision.copy()

            # Adjust stop loss based on risk model
            if ml_predictions['risk_score'] > 0.8:
                # Tighter stop loss for high-risk situations
                adjusted['stop_loss'] = self._calculate_tighter_stop_loss(
                    adjusted['entry_price'],
                    adjusted['stop_loss']
                )

            # Adjust take profit based on pattern recognition
            if ml_predictions['pattern_probability'] > 0.8:
                # More aggressive take profit for strong patterns
                adjusted['take_profit'] = self._calculate_extended_take_profit(
                    adjusted['entry_price'],
                    adjusted['take_profit']
                )

            return adjusted

        except Exception as e:
            self.logger.error(f"Error adjusting trade parameters: {e}")
            return gemini_decision

    async def check_trade_completion(self):
        """Check if the current trade is complete."""
        if self.current_trade is None:
            return True

        symbol = self.current_trade.get("symbol")
        if not symbol:
            self.logger.error("No symbol found in current trade")
            return True

        return await self.trade_executor.check_trade_completion(
            symbol, self.current_trade, self.notification_manager
        )

    async def find_best_opportunity(self, trading_pairs):
        """Find the best trading opportunity using ML models."""
        try:
            best_opportunity = None
            highest_score = 0

            for symbol in trading_pairs:
                market_data = await self.binance_client.get_market_data(symbol)
                if not market_data:
                    continue

                # Get ML predictions
                ml_predictions = self.trading_model.predict({
                    'market_data': market_data,
                    'symbol': symbol,
                    'timeframe': '1m'
                })

                if not ml_predictions:
                    continue

                # Get Gemini decision
                gemini_decision = self.gemini_client.get_trading_decision(
                    closing_prices=market_data["closing_prices"],
                    volumes=market_data["volumes"],
                    high_prices=market_data["high_prices"],
                    low_prices=market_data["low_prices"],
                    open_prices=market_data["open_prices"],
                    symbol=symbol
                )

                if gemini_decision['decision'] == 'waiting':
                    continue

                # Combine ML and Gemini scores
                combined_score = (
                    ml_predictions['combined_score'] * 0.6 +  # ML models weight
                    self._calculate_opportunity_score(gemini_decision) * 0.4  # Gemini weight
                )

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_opportunity = {
                        'symbol': symbol,
                        'score': combined_score,
                        'decision_data': gemini_decision,
                        'ml_predictions': ml_predictions
                    }

            return best_opportunity

        except Exception as e:
            self.logger.error(f"Error finding best opportunity: {e}")
            return None

    async def execute_best_opportunity(self, opportunity):
        """Execute the best trading opportunity."""
        if not opportunity or "symbol" not in opportunity or "decision_data" not in opportunity:
            self.logger.error("Invalid opportunity data")
            await self.telegram_bot.send_message("⚠️ *Invalid Opportunity Data*: Unable to execute trade.")
            return False

        symbol = opportunity["symbol"]
        decision = opportunity["decision_data"]

        # Process and execute the trading decision
        trade_result = await self.process_trading_decision(symbol, decision)
        
        if trade_result["decision"] != "waiting":
            self.current_trade = {
                "symbol": symbol,
                "entry_price": decision["entry_price"],
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"]
            }
            # Notify about the new trade
            await self.notify_trade_opened(symbol, decision)
            return True

        await self.telegram_bot.send_message(f"⚠️ *Trade Not Executed*: {trade_result['reason']}")
        return False

    async def notify_trade_opened(self, symbol, decision):
        """Send notification when a new trade is opened."""
        await self.notification_manager.notify_trade_opened(symbol, decision)

    async def notify_trade_closed(self, symbol, result, entry_price, exit_price):
        """Send notification when a trade is closed."""
        await self.notification_manager.notify_trade_closed(
            symbol, result, entry_price, exit_price, self.performance_tracker
        )

    async def on_trade_completed(self, trade_result):
        """Update ML models after trade completion."""
        try:
            # Add trade to history
            self.trade_analyzer.analyze_trade(trade_result)

            # Train models incrementally with updated trade history
            self.trading_model.train(self.trade_analyzer.trade_history)
            await self.telegram_bot.send_message("📊 *Model Retrained*: Updated with the latest trade data.")

        except Exception as e:
            self.logger.error(f"Error updating ML models: {e}")
            await self.telegram_bot.send_message(f"❌ *Model Update Error*: {e}")

    def _calculate_opportunity_score(self, gemini_decision):
        """Calculate the opportunity score based on Gemini's decision data."""
        try:
            # Use confidence and trend strength to calculate the score
            confidence = gemini_decision.get("confidence", 0)
            trend_strength = gemini_decision.get("trend_strength", "weak")

            # Assign numerical values to trend strength
            trend_strength_score = {
                "weak": 0.5,
                "moderate": 1.0,
                "strong": 1.5
            }.get(trend_strength, 0.5)

            # Calculate the opportunity score
            opportunity_score = confidence * trend_strength_score
            return opportunity_score
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0