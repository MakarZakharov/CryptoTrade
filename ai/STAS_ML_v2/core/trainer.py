"""
Main training orchestrator for STAS_ML v2

Coordinates all aspects of model training, validation, and evaluation.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .config import Config
from .base import BaseModel, ModelFactory, Logger, MetricsCalculator
from ..data.processor import DataProcessor
from ..validation.validator import ModelValidator
from ..backtesting.engine import BacktestEngine


class ModelTrainer:
    """Main class for orchestrating model training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger("ModelTrainer", "INFO" if config.verbose else "WARNING")
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.validator = ModelValidator(config)
        self.backtest_engine = BacktestEngine(config)
        
        # Training state
        self.model = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.training_results = {}
        
        # Setup directories
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized trainer for experiment: {config.experiment_name}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        self.logger.info("Preparing data...")
        
        # Load and process data
        raw_data = self.data_processor.load_data()
        processed_data = self.data_processor.process_data(raw_data)
        
        # Create features and targets
        X, y = self.data_processor.create_features_and_targets(processed_data)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(X, y)
        
        # Store for later use
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        self.logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        self.logger.info(f"Features: {X_train.shape[1]}, Target type: {self.config.model.target_type.value}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self) -> Dict[str, Any]:
        """Train the model."""
        if self.training_data is None:
            self.prepare_data()
        
        X_train, y_train = self.training_data
        X_val, y_val = self.validation_data
        
        self.logger.info(f"Training {self.config.model.model_type.value} model...")
        
        # Create model
        self.model = ModelFactory.create_model(self.config)
        
        # Record training start time
        training_start = time.time()
        
        # Train model
        training_metrics = self.model.fit(X_train, y_train, X_val, y_val)
        
        # Record training time
        training_time = time.time() - training_start
        training_metrics['training_time_seconds'] = training_time
        
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Store training results
        self.training_results.update(training_metrics)
        
        return training_metrics
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before validation")
        
        self.logger.info("Validating model...")
        
        X_train, y_train = self.training_data
        X_val, y_val = self.validation_data
        X_test, y_test = self.test_data
        
        # Cross-validation on training + validation data
        X_cv = np.vstack([X_train, X_val])
        y_cv = np.hstack([y_train, y_val])
        cv_results = self.validator.cross_validate(self.model, X_cv, y_cv)
        
        # Test set evaluation
        test_results = self.validator.validate(self.model, X_test, y_test)
        
        # Calculate additional metrics
        y_pred = self.model.predict(X_test)
        y_proba = None
        if self.model.is_classification():
            try:
                y_proba = self.model.predict_proba(X_test)
            except:
                pass
        
        # Get appropriate metrics
        if self.model.is_classification():
            test_metrics = MetricsCalculator.classification_metrics(y_test, y_pred, y_proba)
        else:
            test_metrics = MetricsCalculator.regression_metrics(y_test, y_pred)
        
        validation_results = {
            'cross_validation': cv_results,
            'test_evaluation': test_results,
            'test_metrics': test_metrics
        }
        
        self.training_results.update(validation_results)
        
        self.logger.info(f"Model validation completed")
        
        # Log key metrics
        if self.model.is_classification():
            self.logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"Test F1-Score: {test_metrics.get('f1', 0):.4f}")
        else:
            self.logger.info(f"Test RMSE: {test_metrics.get('rmse', 0):.6f}")
            self.logger.info(f"Test R²: {test_metrics.get('r2', 0):.4f}")
        
        return validation_results
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run backtesting."""
        if self.model is None:
            raise ValueError("Model must be trained before backtesting")
        
        self.logger.info("Running backtest...")
        
        # Get full processed data for backtesting
        raw_data = self.data_processor.load_data()
        processed_data = self.data_processor.process_data(raw_data)
        
        # Run backtest
        backtest_results = self.backtest_engine.run_backtest(self.model, processed_data)
        
        self.training_results['backtest'] = backtest_results
        
        self.logger.info("Backtest completed")
        
        # Log key results
        if 'total_return' in backtest_results:
            self.logger.info(f"Total Return: {backtest_results['total_return']:.2%}")
            self.logger.info(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
            self.logger.info(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.4f}")
        
        return backtest_results
    
    def train_complete_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline."""
        self.logger.info("Starting complete training pipeline...")
        
        pipeline_start = time.time()
        
        try:
            # 1. Prepare data
            self.prepare_data()
            
            # 2. Train model
            training_results = self.train_model()
            
            # 3. Validate model
            validation_results = self.validate_model()
            
            # 4. Run backtest
            backtest_results = self.run_backtest()
            
            # 5. Compile final results
            pipeline_time = time.time() - pipeline_start
            
            final_results = {
                'experiment_name': self.config.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'pipeline_time_seconds': pipeline_time,
                'config': self.config.to_dict(),
                'training': training_results,
                'validation': validation_results,
                'backtest': backtest_results,
                'status': 'completed'
            }
            
            # 6. Save results
            self.save_results(final_results)
            
            # 7. Save model
            model_path = self.save_model()
            final_results['model_path'] = str(model_path)
            
            self.logger.info(f"Complete pipeline finished in {pipeline_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            error_results = {
                'experiment_name': self.config.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
            self.save_results(error_results)
            raise
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results."""
        results_path = self.experiment_dir / "results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        try:
            # Flatten nested results for CSV
            flat_results = self._flatten_dict(results)
            df = pd.DataFrame([flat_results])
            csv_path = self.experiment_dir / "results.csv"
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.logger.warning(f"Could not save CSV results: {e}")
        
        self.logger.info(f"Results saved to {results_path}")
    
    def save_model(self) -> Path:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = self.experiment_dir / "model"
        model_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = model_path / "model.joblib"
        self.model.save(str(model_file))
        
        # Save feature names and preprocessing info
        preprocessing_info = {
            'feature_names': self.data_processor.get_feature_names(),
            'preprocessing_params': self.data_processor.get_preprocessing_params(),
            'config': self.config.to_dict()
        }
        
        info_file = model_path / "preprocessing_info.json"
        with open(info_file, 'w') as f:
            json.dump(preprocessing_info, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> BaseModel:
        """Load a trained model."""
        model_path = Path(model_path)
        
        # Load preprocessing info
        info_file = model_path / "preprocessing_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                preprocessing_info = json.load(f)
            
            # Update data processor with saved parameters
            self.data_processor.set_preprocessing_params(preprocessing_info['preprocessing_params'])
        
        # Create and load model
        model = ModelFactory.create_model(self.config)
        model_file = model_path / "model.joblib"
        model.load(str(model_file))
        
        self.model = model
        self.logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Process data
        processed_data = self.data_processor.process_data(data)
        X, _ = self.data_processor.create_features_and_targets(processed_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        if not self.model.is_classification():
            raise ValueError("Probabilities only available for classification models")
        
        # Process data
        processed_data = self.data_processor.process_data(data)
        X, _ = self.data_processor.create_features_and_targets(processed_data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if self.model is None:
            return None
        
        importance = self.model.get_feature_importance()
        if importance is None:
            return None
        
        # Map to feature names
        feature_names = self.data_processor.get_feature_names()
        if len(feature_names) == len(importance):
            return dict(zip(feature_names, importance.values()))
        
        return importance
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        if not self.training_results:
            return {}
        
        summary = {
            'experiment_name': self.config.experiment_name,
            'model_type': self.config.model.model_type.value,
            'target_type': self.config.model.target_type.value,
            'data_symbol': self.config.data.symbol,
            'training_time': self.training_results.get('training_time_seconds', 0),
        }
        
        # Add key metrics
        if 'test_metrics' in self.training_results:
            test_metrics = self.training_results['test_metrics']
            if self.model.is_classification():
                summary.update({
                    'test_accuracy': test_metrics.get('accuracy', 0),
                    'test_f1': test_metrics.get('f1', 0),
                    'test_precision': test_metrics.get('precision', 0),
                    'test_recall': test_metrics.get('recall', 0)
                })
            else:
                summary.update({
                    'test_rmse': test_metrics.get('rmse', 0),
                    'test_r2': test_metrics.get('r2', 0),
                    'test_mae': test_metrics.get('mae', 0)
                })
        
        # Add backtest results
        if 'backtest' in self.training_results:
            backtest = self.training_results['backtest']
            summary.update({
                'backtest_total_return': backtest.get('total_return', 0),
                'backtest_max_drawdown': backtest.get('max_drawdown', 0),
                'backtest_sharpe_ratio': backtest.get('sharpe_ratio', 0),
                'backtest_win_rate': backtest.get('win_rate', 0)
            })
        
        return summary


def quick_train(symbol: str = "BTCUSDT", 
               model_type: str = "xgboost",
               target_type: str = "direction") -> ModelTrainer:
    """Quick training with default configuration."""
    from .config import create_default_config
    
    config = create_default_config(symbol, model_type, target_type)
    trainer = ModelTrainer(config)
    trainer.train_complete_pipeline()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    trainer = quick_train("BTCUSDT", "xgboost", "direction")
    summary = trainer.get_training_summary()
    print(json.dumps(summary, indent=2))