import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    def __init__(self, data_dir="/home/thomas/tradingbot/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.trade_history_file = self.data_dir / "trade_history.json"
        self.trade_patterns_file = self.data_dir / "trade_patterns.json"
        self.load_historical_data()
        self.smc_patterns = {
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_levels': [],
            'break_of_structure': [],
            'change_of_character': [],
            'institutional_moves': []
        }
        
    def load_historical_data(self):
        """Load historical trade data and patterns."""
        try:
            if self.trade_history_file.exists():
                with open(self.trade_history_file, 'r') as f:
                    self.trade_history = json.load(f)
            else:
                self.trade_history = []

            if self.trade_patterns_file.exists():
                with open(self.trade_patterns_file, 'r') as f:
                    self.patterns = json.load(f)
            else:
                self.patterns = {
                    'entry_patterns': [],
                    'exit_patterns': [],
                    'risk_patterns': [],
                    'market_patterns': []
                }
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.trade_history = []
            self.patterns = {'entry_patterns': [], 'exit_patterns': [], 'risk_patterns': [], 'market_patterns': []}

    def analyze_trade(self, trade_data):
        """Analyze a completed trade and extract patterns."""
        try:
            if 'pnl_percent' not in trade_data:
                logger.info("Trade data missing 'pnl_percent'. Estimating value.")
                trade_data['pnl_percent'] = self._estimate_pnl_percent(trade_data)

            # Add trade to history
            self.trade_history.append(trade_data)
            
            # Extract patterns from successful trades
            if trade_data['pnl_percent'] > 0:
                self._extract_entry_pattern(trade_data)
                self._extract_exit_pattern(trade_data)
                self._extract_risk_pattern(trade_data)
                self._extract_market_pattern(trade_data)
            
            # Save updated data
            self._save_data()
            
            # Return optimized parameters for next trade
            return self._generate_optimized_params()
        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")
            return None

    def _estimate_pnl_percent(self, trade_data):
        """Estimate pnl_percent based on entry and exit prices."""
        entry_price = trade_data.get('entry_price', 1)
        exit_price = trade_data.get('exit_price', 1)
        return ((exit_price - entry_price) / entry_price) * 100

    def _extract_entry_pattern(self, trade):
        """Extract successful entry patterns."""
        pattern = {
            'market_structure': trade.get('market_structure', {}),
            'entry_price': trade['entry_price'],
            'success_rate': self._calculate_success_rate(trade),
            'volume_profile': trade.get('volume_profile', {}),
            'timestamp': datetime.now().isoformat()
        }
        self.patterns['entry_patterns'].append(pattern)

    def _extract_exit_pattern(self, trade):
        """Extract successful exit patterns."""
        pattern = {
            'take_profit_hit': trade['exit_price'] >= trade.get('take_profit', 0),
            'exit_price_to_atr': trade.get('exit_price_to_atr', 0),
            'market_conditions': trade.get('market_conditions', {}),
            'hold_time': trade.get('hold_time', 0),
            'timestamp': datetime.now().isoformat()
        }
        self.patterns['exit_patterns'].append(pattern)

    def _generate_optimized_params(self):
        """Generate optimized trading parameters based on historical patterns."""
        if not self.trade_history:
            return None

        successful_trades = [t for t in self.trade_history if t['pnl_percent'] > 0]
        if not successful_trades:
            return None

        # Calculate optimal parameters
        optimal_params = {
            'risk_reward_ratio': self._calculate_optimal_risk_reward(),
            'entry_conditions': self._get_best_entry_conditions(),
            'exit_conditions': self._get_best_exit_conditions(),
            'stop_loss_calculation': self._optimize_stop_loss(),
            'take_profit_levels': self._optimize_take_profit_levels(),
            'market_conditions': self._get_optimal_market_conditions()
        }

        # Generate optimized Gemini prompt
        prompt_template = self._generate_dynamic_prompt(optimal_params)
        
        return {
            'optimal_params': optimal_params,
            'prompt_template': prompt_template
        }

    def _calculate_optimal_risk_reward(self):
        """Calculate optimal risk-reward ratio based on successful trades."""
        successful_trades = [t for t in self.trade_history if t['pnl_percent'] > 0]
        if not successful_trades:
            return 2.0  # Default

        risk_rewards = [t.get('risk_reward_ratio', 2.0) for t in successful_trades]
        return np.mean(risk_rewards)

    def _get_best_entry_conditions(self):
        """Identify most successful entry conditions."""
        successful_entries = [p for p in self.patterns['entry_patterns'] 
                            if p['success_rate'] > 0.7]
        if not successful_entries:
            return {}

        return {
            'price_action': self._analyze_price_patterns(successful_entries),
            'volume_profile': self._analyze_volume_patterns(successful_entries),
            'market_structure': self._analyze_structure_patterns(successful_entries)
        }

    def _save_data(self):
        """Save updated trade history and patterns."""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            with open(self.trade_patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")

    def _generate_dynamic_prompt(self, optimal_params):
        """Generate dynamic prompt with learned SMC patterns."""
        smc_insights = self._get_smc_insights()
        
        prompt = f"""
        ## ADVANCED SMC TRADING ANALYSIS
        
        Based on {len(self.trade_history)} analyzed trades:
        
        1. Institutional Patterns:
        - Order Blocks: {smc_insights['order_blocks']}
        - Liquidity Levels: {smc_insights['liquidity_levels']}
        - Break of Structure: {smc_insights['bos_patterns']}
        - Change of Character: {smc_insights['choch_patterns']}
        
        2. Entry Conditions:
        - Premium/Discount Zones: {optimal_params['entry_zones']}
        - Volume Profile: {optimal_params['volume_analysis']}
        - Market Structure: {optimal_params['market_structure']}
        
        3. Risk Management:
        - Stop Loss: Place behind valid order blocks
        - Take Profit: At opposing liquidity levels
        - Position Size: Based on risk/volatility profile
        
        4. Execution Rules:
        - Wait for institutional confirmation
        - Monitor order block validity
        - Track liquidity sweeps
        """
        return prompt

    def _analyze_price_patterns(self, successful_entries):
        """Analyze successful price action patterns."""
        patterns = {
            'order_blocks': self._identify_order_blocks(),
            'liquidity_sweeps': self._identify_liquidity_sweeps(),
            'bos_patterns': self._identify_break_of_structure(),
            'choch_patterns': self._identify_change_of_character(),
            'fvg_patterns': self._identify_fair_value_gaps()
        }
        
        # Learn from successful patterns
        self._learn_institutional_patterns(patterns)
        return patterns

    def _identify_order_blocks(self):
        """Identify institutional order blocks."""
        order_blocks = []
        for trade in self.trade_history:
            if trade['pnl_percent'] > 2.0:  # Focus on very profitable trades
                candles = trade.get('pre_entry_candles', [])
                for i in range(len(candles)-3):
                    # Look for orderblock characteristics
                    if self._is_orderblock_formation(candles[i:i+3]):
                        order_blocks.append({
                            'price_level': candles[i]['close'],
                            'volume': candles[i]['volume'],
                            'success_rate': self._calculate_ob_success_rate(candles[i]['close']),
                            'type': 'bullish' if trade['direction'] == 'buy' else 'bearish'
                        })
        return order_blocks

    def _identify_liquidity_sweeps(self):
        """Identify institutional liquidity sweeps."""
        sweeps = []
        for trade in self.trade_history:
            if trade['pnl_percent'] > 0:
                pre_entry = trade.get('pre_entry_data', {})
                if self._is_liquidity_sweep(pre_entry):
                    sweeps.append({
                        'level': trade['entry_price'],
                        'success_rate': trade['pnl_percent'],
                        'volume_spike': pre_entry.get('volume_spike', False),
                        'reversal_quality': self._calculate_reversal_quality(pre_entry)
                    })
        return sweeps

    def _learn_institutional_patterns(self, new_patterns):
        """Learn and adapt to institutional trading patterns."""
        try:
            # Update success rates for each pattern type
            for pattern_type, patterns in new_patterns.items():
                if not patterns:
                    continue
                    
                # Calculate success metrics
                success_rate = sum(p.get('success_rate', 0) for p in patterns) / len(patterns)
                
                # Update pattern weights based on success
                if success_rate > 0.7:  # High success rate
                    self.pattern_weights[pattern_type] *= 1.1  # Increase importance
                elif success_rate < 0.3:  # Low success rate
                    self.pattern_weights[pattern_type] *= 0.9  # Decrease importance
                    
                # Store learned patterns
                self.smc_patterns[pattern_type].extend([
                    p for p in patterns 
                    if p.get('success_rate', 0) > 0.5  # Only store successful patterns
                ])
                
            # Cleanup old patterns
            self._cleanup_old_patterns()
            
        except Exception as e:
            logger.error(f"Error learning institutional patterns: {e}")
