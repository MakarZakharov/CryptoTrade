#!/usr/bin/env python3
"""
Simple and clean training script for STAS_ML models.
Focused on practical ML training without unnecessary complexity.
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.ml_config import MLConfig
from CryptoTrade.ai.STAS_ML.training.trainer import MLTrainer


def create_simple_config(symbol='BTCUSDT', model_type='xgboost', target_type='direction'):
    """Create a simple, optimized configuration for training."""
    return MLConfig(
        symbol=symbol,
        timeframe='1d',
        model_type=model_type,
        target_type=target_type,
        lookback_window=30,
        min_price_change_threshold=0.02,  # 2% minimum change
        signal_confidence_threshold=0.6   # 60% confidence
    )


def quick_train(symbol='BTCUSDT', model_type='xgboost', target_type='direction', name=None):
    """Simple, fast training function."""
    print(f"🚀 Starting ML Training")
    print(f"Symbol: {symbol}, Model: {model_type}, Target: {target_type}")
    print("=" * 50)
    
    # Create configuration
    config = create_simple_config(symbol, model_type, target_type)
    
    # Create trainer
    trainer = MLTrainer(config, custom_model_name=name)
    
    try:
        # Train model
        metrics = trainer.train()
        
        # Show results
        print(f"\n✅ Training completed!")
        print(f"Experiment: {trainer.experiment_name}")
        
        if config.target_type == 'direction':
            print(f"Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        else:
            print(f"Test RMSE: {metrics.get('test_rmse', 0):.6f}")
        
        # Trading results if available
        if 'trading_total_return_pct' in metrics:
            print(f"Trading Return: {metrics['trading_total_return_pct']:+.2f}%")
            print(f"Win Rate: {metrics['trading_win_rate']*100:.1f}%")
        
        # Save model
        model_path = trainer.save_model()
        print(f"📁 Model saved: {model_path}")
        
        return trainer, metrics
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, {}


def main():
    """Main training function with simple interface."""
    print("🤖 STAS_ML Simple Training")
    print("=" * 40)
    
    # Default parameters
    symbol = input("Symbol (default: BTCUSDT): ").strip() or 'BTCUSDT'
    model_type = input("Model type (xgboost/random_forest, default: xgboost): ").strip() or 'xgboost'
    target_type = input("Target type (direction/price_change, default: direction): ").strip() or 'direction'
    
    # Train model
    trainer, metrics = quick_train(symbol, model_type, target_type)
    
    if trainer:
        print("\n🎉 Training successful!")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()