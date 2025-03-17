from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self):
        self.pattern_classifier = RandomForestClassifier(n_estimators=100)
        self.entry_classifier = GradientBoostingClassifier()
        self.risk_model = GradientBoostingClassifier()
        self.scaler = StandardScaler()
        self.models_path = "/home/thomas/tradingbot/data/models/"
        self.training_data_file = "/home/thomas/tradingbot/data/training_data.json"
        self.training_data = self._load_training_data()
        # Create models directory if it doesn't exist
        Path(self.models_path).mkdir(parents=True, exist_ok=True)
        self.is_fitted = False
        
        # Initialize with dummy balanced dataset
        self._initialize_with_dummy_data()

    def _initialize_with_dummy_data(self):
        """Initialize models with balanced dummy data."""
        dummy_data = [
            # Profitable trades
            {'pnl_percent': 2.5, 'hold_time': 300, 'volume_profile': {'relative_volume': 1.5}, 'market_structure': {'trend_strength': 0.9}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 1}, 'entry_quality': 0.85},
            {'pnl_percent': 3.0, 'hold_time': 400, 'volume_profile': {'relative_volume': 1.8}, 'market_structure': {'trend_strength': 0.95}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 1}, 'entry_quality': 0.9},
            {'pnl_percent': 1.8, 'hold_time': 250, 'volume_profile': {'relative_volume': 1.2}, 'market_structure': {'trend_strength': 0.8}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 1}, 'entry_quality': 0.75},
            # Losing trades
            {'pnl_percent': -1.5, 'hold_time': 200, 'volume_profile': {'relative_volume': 0.7}, 'market_structure': {'trend_strength': 0.4}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.4},
            {'pnl_percent': -2.0, 'hold_time': 180, 'volume_profile': {'relative_volume': 0.6}, 'market_structure': {'trend_strength': 0.3}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.35},
            {'pnl_percent': -0.8, 'hold_time': 150, 'volume_profile': {'relative_volume': 0.9}, 'market_structure': {'trend_strength': 0.5}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.5},
            # Neutral trades
            {'pnl_percent': 0.0, 'hold_time': 100, 'volume_profile': {'relative_volume': 1.0}, 'market_structure': {'trend_strength': 0.6}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.6},
            {'pnl_percent': 0.1, 'hold_time': 120, 'volume_profile': {'relative_volume': 1.1}, 'market_structure': {'trend_strength': 0.7}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.65},
            # Extreme cases
            {'pnl_percent': 5.0, 'hold_time': 600, 'volume_profile': {'relative_volume': 2.0}, 'market_structure': {'trend_strength': 1.0}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 1}, 'entry_quality': 0.95},
            {'pnl_percent': -3.0, 'hold_time': 300, 'volume_profile': {'relative_volume': 0.5}, 'market_structure': {'trend_strength': 0.2}, 'pre_entry_candles': [], 'market_conditions': {'liquidity_sweep': 0}, 'entry_quality': 0.2}
        ]
        
        try:
            # Ensure balanced classes for each model
            pattern_labels = [1 if t['pnl_percent'] > 0 else 0 for t in dummy_data]
            entry_labels = [1 if t['entry_quality'] > 0.7 else 0 for t in dummy_data]
            risk_labels = [1 if t['pnl_percent'] > 1.5 else 0 for t in dummy_data]
            
            # Check class balance
            if len(set(pattern_labels)) < 2:
                dummy_data.append(self._create_opposite_example(dummy_data[0], 'pattern'))
            if len(set(entry_labels)) < 2:
                dummy_data.append(self._create_opposite_example(dummy_data[0], 'entry'))
            if len(set(risk_labels)) < 2:
                dummy_data.append(self._create_opposite_example(dummy_data[0], 'risk'))

            features = self.prepare_features(dummy_data)
            
            # Update labels after potential additions
            pattern_labels = [1 if t['pnl_percent'] > 0 else 0 for t in dummy_data]
            entry_labels = [1 if t['entry_quality'] > 0.7 else 0 for t in dummy_data]
            risk_labels = [1 if t['pnl_percent'] > 1.5 else 0 for t in dummy_data]

            # Scale and fit
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            
            # Train models
            self.pattern_classifier.fit(features_scaled, pattern_labels)
            self.entry_classifier.fit(features_scaled, entry_labels)
            self.risk_model.fit(features_scaled, risk_labels)
            
            self.is_fitted = True
            logger.info("Models initialized with diverse dummy data")
            
        except Exception as e:
            logger.error(f"Error initializing models with dummy data: {e}")
            self.is_fitted = False

    def _create_opposite_example(self, example, model_type, target_class=None):
        """Create an opposite class example for balancing."""
        opposite = example.copy()
        if model_type == 'pattern':
            if target_class == 0:
                opposite['pnl_percent'] = -abs(example['pnl_percent'])  # Ensure negative pnl
            elif target_class == 1:
                opposite['pnl_percent'] = abs(example['pnl_percent'])  # Ensure positive pnl
        elif model_type == 'entry':
            opposite['entry_quality'] = 0.3 if example['entry_quality'] > 0.7 else 0.8
        elif model_type == 'risk':
            opposite['pnl_percent'] = 1.0 if example['pnl_percent'] > 1.5 else 2.0
        return opposite

    def prepare_features(self, trade_data):
        """Extract features from trade data."""
        features = []
        for trade in trade_data:
            feature_vector = [
                trade.get('pnl_percent', 0),
                trade.get('hold_time', 0),
                trade.get('volume_profile', {}).get('relative_volume', 1),
                trade.get('market_structure', {}).get('trend_strength', 0),
                *self._extract_candlestick_features(trade.get('pre_entry_candles', [])),
                *self._extract_institutional_features(trade)
            ]
            if any(feature_vector):  # Ensure the feature vector is not empty
                features.append(feature_vector)
        if not features:
            logger.error("No valid features extracted from trade data.")
        return np.array(features)

    def _extract_candlestick_features(self, candles):
        """Extract features from candlestick patterns."""
        if not candles:
            return [0] * 5  # Return zeros if no candles
            
        returns = np.diff([c['close'] for c in candles])
        volumes = [c['volume'] for c in candles]
        
        return [
            np.mean(returns),
            np.std(returns),
            np.mean(volumes),
            np.max(returns),
            np.min(returns)
        ]

    def _extract_institutional_features(self, trade):
        """Extract institutional trading features."""
        return [
            trade.get('market_conditions', {}).get('liquidity_sweep', 0),
            trade.get('market_conditions', {}).get('order_block_strength', 0),
            trade.get('market_conditions', {}).get('fvg_presence', 0),
            trade.get('market_conditions', {}).get('bos_confirmed', 0),
            trade.get('market_conditions', {}).get('choch_confirmed', 0)
        ]

    def _load_training_data(self):
        """Load training data from a file."""
        try:
            if Path(self.training_data_file).exists():
                with open(self.training_data_file, "r") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return []

    def _save_training_data(self):
        """Save training data to a file."""
        try:
            with open(self.training_data_file, "w") as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Training data saved to {self.training_data_file}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def train(self, trade_history):
        """Train all models on historical trade data."""
        if not trade_history:
            logger.error("No trade history provided for training.")
            return False

        try:
            # Append new trade history to the existing training data
            self.training_data.extend(trade_history)
            self._save_training_data()

            # Prepare features and labels
            features = self.prepare_features(self.training_data)
            pattern_labels = [1 if t['pnl_percent'] > 0 else 0 for t in self.training_data]

            # Log class distribution
            class_counts = {label: pattern_labels.count(label) for label in set(pattern_labels)}
            logger.info(f"Class distribution: {class_counts}")

            # Proceed with training even if classes are imbalanced
            if len(set(pattern_labels)) < 2:
                logger.warning("Imbalanced classes detected. Training may be suboptimal.")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, pattern_labels, test_size=0.2, random_state=42
            )

            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train models
            self.pattern_classifier.fit(X_train_scaled, y_train)
            entry_labels = [1 if t.get('entry_quality', 0) > 0.7 else 0 for t in self.training_data]
            self.entry_classifier.fit(X_train_scaled, entry_labels[:len(X_train)])
            risk_labels = [1 if t['pnl_percent'] > 2 else 0 for t in self.training_data]
            self.risk_model.fit(X_train_scaled, risk_labels[:len(X_train)])

            self.is_fitted = True
            logger.info("Models trained successfully")
            return True

        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False

    def predict(self, current_market_data):
        """Make predictions using trained models."""
        try:
            if not self.is_fitted:
                logger.warning("Models not fitted, using default predictions")
                return self._get_default_predictions()

            # Prepare features for prediction
            features = self.prepare_features([current_market_data])
            if len(features) == 0:
                logger.error("No valid features extracted from market data. Using default predictions.")
                return self._get_default_predictions()

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Handle single sample prediction
            try:
                pattern_prob = self.pattern_classifier.predict_proba(features_scaled)[0][1]
                entry_prob = self.entry_classifier.predict_proba(features_scaled)[0][1]
                risk_prob = self.risk_model.predict_proba(features_scaled)[0][1]
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return self._get_default_predictions()

            # Combine predictions into a single score
            predictions = {
                'pattern_probability': pattern_prob,
                'entry_quality': entry_prob,
                'risk_score': risk_prob,
                'combined_score': (
                    pattern_prob * 0.4 +
                    entry_prob * 0.4 +
                    risk_prob * 0.2
                )
            }
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return self._get_default_predictions()

    def _get_default_predictions(self):
        """Return default prediction values."""
        return {
            'pattern_probability': 0.5,
            'entry_quality': 0.5,
            'risk_score': 0.5,
            'combined_score': 0.5
        }

    def save_models(self):
        """Save trained models to disk."""
        try:
            joblib.dump(self.pattern_classifier, f"{self.models_path}pattern_classifier.pkl")
            joblib.dump(self.entry_classifier, f"{self.models_path}entry_classifier.pkl")
            joblib.dump(self.risk_model, f"{self.models_path}risk_model.pkl")
            joblib.dump(self.scaler, f"{self.models_path}scaler.pkl")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk or initialize new ones."""
        try:
            if all(Path(f"{self.models_path}{f}").exists() for f in 
                ["pattern_classifier.pkl", "entry_classifier.pkl", 
                 "risk_model.pkl", "scaler.pkl"]):
                # Load existing models
                self.pattern_classifier = joblib.load(f"{self.models_path}pattern_classifier.pkl")
                self.entry_classifier = joblib.load(f"{self.models_path}entry_classifier.pkl")
                self.risk_model = joblib.load(f"{self.models_path}risk_model.pkl")
                self.scaler = joblib.load(f"{self.models_path}scaler.pkl")
                return True
            else:
                logger.info("No existing models found, will train new ones")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
