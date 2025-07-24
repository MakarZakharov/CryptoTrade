"""
Training script for STAS_ML v2

Simple, clean interface for training ML models for cryptocurrency trading.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CryptoTrade.ai.STAS_ML_v2.core.config import Config, create_default_config
from CryptoTrade.ai.STAS_ML_v2.core.trainer import ModelTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='STAS_ML v2 Training')
    
    # Data arguments
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1d', help='Timeframe')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date')
    parser.add_argument('--end-date', default='2024-12-31', help='End date')
    
    # Model arguments
    parser.add_argument('--model', default='xgboost', 
                       choices=['xgboost', 'random_forest', 'lstm', 'ensemble', 'linear'],
                       help='Model type')
    parser.add_argument('--target', default='direction',
                       choices=['direction', 'price_change', 'volatility'],
                       help='Target type')
    
    # Training arguments
    parser.add_argument('--experiment-name', help='Custom experiment name')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--quick', action='store_true', help='Quick training with minimal validation')
    
    # Output arguments
    parser.add_argument('--output-dir', default='experiments', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            config = Config.from_file(args.config)
        else:
            config = create_default_config(args.symbol, args.model, args.target)
        
        # Override config with command line arguments
        config.data.symbol = args.symbol
        config.data.timeframe = config.data.timeframe.__class__(args.timeframe)
        config.data.start_date = args.start_date
        config.data.end_date = args.end_date
        config.model.model_type = config.model.model_type.__class__(args.model)
        config.model.target_type = config.model.target_type.__class__(args.target)
        config.output_dir = args.output_dir
        config.verbose = args.verbose
        
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        
        # Validate configuration
        config.validate()
        
        print("🚀 STAS_ML v2 Training Started")
        print("="*50)
        print(f"Symbol: {config.data.symbol}")
        print(f"Timeframe: {config.data.timeframe.value}")
        print(f"Model: {config.model.model_type.value}")
        print(f"Target: {config.model.target_type.value}")
        print(f"Experiment: {config.experiment_name}")
        print("="*50)
        
        # Create trainer
        trainer = ModelTrainer(config)
        
        if args.quick:
            # Quick training: just train and validate
            trainer.prepare_data()
            trainer.train_model()
            trainer.validate_model()
            
            print("✅ Quick training completed!")
            summary = trainer.get_training_summary()
            
            print("\n📊 Training Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        else:
            # Full pipeline
            results = trainer.train_complete_pipeline()
            
            print("✅ Complete training pipeline finished!")
            print(f"📁 Results saved to: {trainer.experiment_dir}")
            
            # Print summary
            summary = trainer.get_training_summary()
            print("\n📊 Final Results:")
            print("-"*40)
            
            if trainer.model.is_classification():
                print(f"Test Accuracy: {summary.get('test_accuracy', 0):.4f}")
                print(f"Test F1-Score: {summary.get('test_f1', 0):.4f}")
            else:
                print(f"Test RMSE: {summary.get('test_rmse', 0):.6f}")
                print(f"Test R²: {summary.get('test_r2', 0):.4f}")
            
            if 'backtest_total_return' in summary:
                print(f"Backtest Return: {summary['backtest_total_return']:.2%}")
                print(f"Max Drawdown: {summary.get('backtest_max_drawdown', 0):.2%}")
                print(f"Sharpe Ratio: {summary.get('backtest_sharpe_ratio', 0):.4f}")
            
            print("-"*40)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()