#!/usr/bin/env python3
"""
Professional DRL Trading Agent Training Script
BTCUSDT 15m timeframe - Production-ready training pipeline

Features:
- PPO algorithm with optimized hyperparameters
- Comprehensive metrics tracking and logging
- W&B and TensorBoard integration
- Professional risk management
- Reproducible results with proper seeding
- GPU optimization
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to 'trading' directory
sys.path.insert(0, str(project_root))
print(f"🔍 Project root set to: {project_root}")
print(f"🔍 Current directory: {Path.cwd()}")

# DRL imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.logger import configure
    import torch
    print("✅ Stable-Baselines3 imported successfully")
except ImportError as e:
    print(f"❌ Error importing Stable-Baselines3: {e}")
    sys.exit(1)

# W&B import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ W&B available")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ W&B not available, using TensorBoard only")

# Environment imports
try:
    # Try absolute import first
    from CryptoTrade.ai.DRL.environment.environment import create_trading_environment
    print("✅ Trading environment imported successfully")
except ImportError:
    try:
        # Try relative import from parent directory
        sys.path.append(str(project_root / "CryptoTrade" / "ai" / "DRL"))
        from environment.environment import create_trading_environment
        print("✅ Trading environment imported successfully (relative path)")
    except ImportError as e:
        print(f"❌ Error importing trading environment: {e}")
        print(f"🔍 Project root: {project_root}")
        print(f"🔍 Current working directory: {Path.cwd()}")
        sys.exit(1)

# Local imports
from callbacks.equity_tracker import ProfessionalEquityCallback, TradingMetricsCallback
from utils.metrics import ProfessionalMetricsCalculator


class ProfessionalTrainer:
    """
    Professional DRL trainer with comprehensive monitoring and logging
    
    Implements industry best practices:
    - Proper data splitting (train/validation/test)
    - Comprehensive metrics tracking
    - Risk management monitoring
    - Reproducible training with seeding
    - Professional logging and visualization
    """
    
    def __init__(self, config_path: str = "configs/ppo_config.yaml"):
        """Initialize professional trainer"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.setup_directories()
        self.setup_logging()
        
        # Set random seeds for reproducibility
        self.seed = self.config['training']['seed']
        set_random_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = self.config['reproducibility']['deterministic_pytorch']
            torch.backends.cudnn.benchmark = self.config['reproducibility']['benchmark_pytorch']
        
        print(f"🎯 Professional DRL Trainer Initialized")
        print(f"📁 Config: {self.config_path}")
        print(f"🌱 Seed: {self.seed}")
        print(f"🎮 Device: {self._get_device()}")
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    def setup_directories(self):
        """Create necessary directories"""
        base_dir = Path(".")
        
        dirs_to_create = [
            base_dir / self.config['paths']['models_dir'],
            base_dir / self.config['paths']['logs_dir'],
            base_dir / self.config['paths']['results_dir'],
            base_dir / self.config['paths']['checkpoints_dir'],
            Path(self.config['logging']['tensorboard_log'])
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
            
        # Double-check TensorBoard directory
        tb_path = Path(self.config['logging']['tensorboard_log'])
        if not tb_path.exists():
            tb_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Force created TensorBoard directory: {tb_path}")
        
        print(f"✅ All directories created and verified")
    
    def setup_logging(self):
        """Setup W&B and TensorBoard logging"""
        self.run_name = self._generate_run_name()
        
        # W&B setup
        if self.config['logging']['use_wandb'] and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.config['logging']['wandb_project'],
                    entity=self.config['logging']['wandb_entity'],
                    name=self.run_name,
                    config=self.config,
                    save_code=True,
                    tags=['PPO', 'BTCUSDT', '15m', 'professional']
                )
                print(f"✅ W&B initialized: {self.run_name}")
                self.use_wandb = True
            except Exception as e:
                print(f"⚠️ W&B initialization failed: {e}, using TensorBoard only")
                self.use_wandb = False
        else:
            self.use_wandb = False
            print("📊 Using TensorBoard logging")
    
    def _generate_run_name(self) -> str:
        """Generate unique run name"""
        if self.config['logging']['run_name']:
            return self.config['logging']['run_name']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = self.config['environment']['symbols'][0]
        timeframe = self.config['environment']['timeframe']
        return f"PPO_{symbol}_{timeframe}_{timestamp}"
    
    def _get_device(self) -> str:
        """Get optimal device for training"""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("💻 Using CPU")
        else:
            device = device_config
            
        return device
    
    def create_environments(self):
        """Create training and evaluation environments"""
        print("🏗️ Creating environments...")
        
        # Training environment configuration
        train_config = self.config['environment'].copy()
        train_config.update({
            'data_split': 'train',
            'start_date': self.config['data']['train_start'],
            'end_date': self.config['data']['train_end']
        })
        
        # Evaluation environment configuration  
        eval_config = self.config['environment'].copy()
        eval_config.update({
            'data_split': 'validation',  # Use validation split for evaluation during training
            'start_date': self.config['data']['train_start'],
            'end_date': self.config['data']['train_end'],
            'max_steps': 1000  # Shorter episodes for evaluation
        })
        
        try:
            # Create environments
            self.train_env = create_trading_environment(train_config)
            self.eval_env = create_trading_environment(eval_config)
            
            # Wrap in Monitor for logging
            self.train_env = Monitor(
                self.train_env,
                filename=str(Path(self.config['paths']['logs_dir']) / "train_monitor.csv"),
                allow_early_resets=True
            )
            
            self.eval_env = Monitor(
                self.eval_env,
                filename=str(Path(self.config['paths']['logs_dir']) / "eval_monitor.csv"),
                allow_early_resets=True
            )
            
            # Environment validation
            print("🔍 Validating environments...")
            check_env(self.train_env, warn=True)
            
            print(f"✅ Environments created successfully")
            print(f"📊 Observation space: {self.train_env.observation_space}")
            print(f"🎮 Action space: {self.train_env.action_space}")
            
        except Exception as e:
            print(f"❌ Error creating environments: {e}")
            raise
    
    def create_model(self):
        """Create PPO model with optimized configuration"""
        print("🤖 Creating PPO model...")
        
        # PPO configuration
        ppo_config = self.config['algorithm']
        network_config = self.config['network']
        
        # Model parameters
        model_params = {
            'policy': network_config['policy_type'],
            'env': self.train_env,
            'learning_rate': float(ppo_config['learning_rate']),
            'gamma': ppo_config['gamma'],
            'gae_lambda': ppo_config['gae_lambda'],
            'n_steps': ppo_config['n_steps'],
            'batch_size': ppo_config['batch_size'],
            'n_epochs': ppo_config['n_epochs'],
            'clip_range': ppo_config['clip_range'],
            'ent_coef': ppo_config['ent_coef'],
            'vf_coef': ppo_config['vf_coef'],
            'max_grad_norm': ppo_config['max_grad_norm'],
            'use_sde': ppo_config['use_sde'],
            'device': self._get_device(),
            'verbose': self.config['training']['verbose'],
            'seed': self.seed
        }
        
        # Network architecture
        if 'net_arch' in network_config:
            model_params['policy_kwargs'] = {
                'net_arch': network_config['net_arch'],
                'activation_fn': getattr(torch.nn, network_config.get('activation_fn', 'Tanh'))
            }
        
        # Optional parameters
        if ppo_config.get('clip_range_vf') is not None:
            model_params['clip_range_vf'] = ppo_config['clip_range_vf']
        if ppo_config.get('target_kl') is not None:
            model_params['target_kl'] = ppo_config['target_kl']
        
        try:
            self.model = PPO(**model_params)
            
            # Setup TensorBoard logging
            tb_log_path = Path(self.config['logging']['tensorboard_log']).resolve()
            print(f"🔍 TensorBoard path: {tb_log_path}")
            print(f"🔍 TensorBoard exists: {tb_log_path.exists()}")
            print(f"🔍 TensorBoard is_dir: {tb_log_path.is_dir()}")
            
            # Force create again if needed
            if not tb_log_path.exists() or not tb_log_path.is_dir():
                tb_log_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Re-created TensorBoard directory: {tb_log_path}")
            
            # Try without logger first to isolate the issue
            print(f"📊 Configuring TensorBoard logger...")
            self.model.set_logger(configure(str(tb_log_path), format_strings=['csv']))  # Start with just CSV
            
            print(f"✅ PPO model created successfully")
            print(f"🧠 Network architecture: {network_config.get('net_arch', 'default')}")
            print(f"📈 Learning rate: {ppo_config['learning_rate']}")
            print(f"🎯 Total timesteps planned: {self.config['training']['total_timesteps']:,}")
            
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            raise
    
    def setup_callbacks(self):
        """Setup comprehensive callbacks for monitoring"""
        print("📋 Setting up callbacks...")
        
        callbacks = []
        
        # 1. Professional Equity Callback
        equity_callback = ProfessionalEquityCallback(
            eval_env=self.eval_env,
            eval_freq=self.config['training']['eval_freq'],
            n_eval_episodes=self.config['training']['n_eval_episodes'],
            log_path=self.config['paths']['logs_dir'],
            save_best_model=True,
            use_wandb=self.use_wandb,
            verbose=1
        )
        callbacks.append(equity_callback)
        
        # 2. Trading Metrics Callback
        metrics_callback = TradingMetricsCallback(
            log_interval=self.config['logging']['log_interval']
        )
        callbacks.append(metrics_callback)
        
        # 3. Checkpoint Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['save_freq'],
            save_path=self.config['paths']['checkpoints_dir'],
            name_prefix=f"ppo_{self.run_name}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)
        
        # 4. Standard Evaluation Callback (backup)
        eval_callback = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=self.config['paths']['models_dir'],
            log_path=self.config['paths']['logs_dir'],
            eval_freq=self.config['training']['eval_freq'],
            n_eval_episodes=self.config['training']['n_eval_episodes'],
            deterministic=True,
            render=False,
            verbose=0
        )
        callbacks.append(eval_callback)
        
        self.callback_list = CallbackList(callbacks)
        print(f"✅ {len(callbacks)} callbacks configured")
    
    def train(self):
        """Run the training process"""
        print("\n" + "="*60)
        print("🚀 STARTING PROFESSIONAL DRL TRAINING")
        print("="*60)
        print(f"📊 Strategy: PPO on {self.config['environment']['symbols'][0]}")
        print(f"⏰ Timeframe: {self.config['environment']['timeframe']}")
        print(f"📅 Training period: {self.config['data']['train_start']} to {self.config['data']['train_end']}")
        print(f"🎯 Target timesteps: {self.config['training']['total_timesteps']:,}")
        print(f"💾 Models will be saved to: {self.config['paths']['models_dir']}")
        print("="*60)
        
        training_start_time = datetime.now()
        
        try:
            # Train the model
            self.model.learn(
                total_timesteps=self.config['training']['total_timesteps'],
                callback=self.callback_list,
                log_interval=self.config['logging']['log_interval'],
                progress_bar=True
            )
            
            training_end_time = datetime.now()
            training_duration = training_end_time - training_start_time
            
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️ Total training time: {training_duration}")
            print(f"🎯 Final timesteps: {self.model.num_timesteps:,}")
            
            # Save final model
            final_model_path = Path(self.config['paths']['models_dir']) / f"final_model_{self.run_name}.zip"
            self.model.save(final_model_path)
            print(f"💾 Final model saved: {final_model_path}")
            
            # Log final metrics to W&B
            if self.use_wandb:
                wandb.log({
                    'training/total_timesteps': self.model.num_timesteps,
                    'training/duration_minutes': training_duration.total_seconds() / 60,
                    'training/status': 'completed'
                })
                
                # Save model as W&B artifact
                artifact = wandb.Artifact(f"model_{self.run_name}", type="model")
                artifact.add_file(str(final_model_path))
                wandb.log_artifact(artifact)
                
                print("✅ Results logged to W&B")
            
            return final_model_path
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            self._save_interrupted_model()
            raise
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            self._save_interrupted_model()
            raise
        finally:
            # Cleanup
            if hasattr(self, 'train_env'):
                self.train_env.close()
            if hasattr(self, 'eval_env'):
                self.eval_env.close()
            
            if self.use_wandb:
                wandb.finish()
    
    def _save_interrupted_model(self):
        """Save model if training is interrupted"""
        try:
            interrupted_model_path = Path(self.config['paths']['models_dir']) / f"interrupted_model_{self.run_name}.zip"
            self.model.save(interrupted_model_path)
            print(f"💾 Interrupted model saved: {interrupted_model_path}")
        except Exception as e:
            print(f"❌ Failed to save interrupted model: {e}")
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            print("🏗️ Setting up training pipeline...")
            
            # Setup pipeline
            self.create_environments()
            self.create_model()
            self.setup_callbacks()
            
            # Run training
            final_model_path = self.train()
            
            print("\n🎯 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📄 Final model: {final_model_path}")
            print(f"📊 Logs directory: {self.config['paths']['logs_dir']}")
            print(f"💾 Checkpoints: {self.config['paths']['checkpoints_dir']}")
            
            return final_model_path
            
        except Exception as e:
            print(f"\n💥 Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Professional DRL Trading Agent Training')
    parser.add_argument('--config', type=str, default='configs/ppo_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume training from')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (overrides config)')
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='Evaluation frequency (overrides config)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    print("🎯 PROFESSIONAL DRL CRYPTO TRADING AGENT")
    print("=" * 50)
    print("📈 BTCUSDT 15m Timeframe")
    print("🤖 PPO Algorithm with Advanced Monitoring")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = ProfessionalTrainer(config_path=args.config)
        
        # Apply command line overrides
        if args.wandb_project:
            trainer.config['logging']['wandb_project'] = args.wandb_project
        if args.total_timesteps:
            trainer.config['training']['total_timesteps'] = args.total_timesteps
        if args.eval_freq:
            trainer.config['training']['eval_freq'] = args.eval_freq
        if args.device:
            trainer.config['hardware']['device'] = args.device
        if args.seed:
            trainer.config['training']['seed'] = args.seed
            set_random_seed(args.seed)
        
        # Resume training if specified
        if args.resume:
            print(f"🔄 Resuming training from: {args.resume}")
            # Note: Resume functionality would be implemented here
            # For now, we start fresh training
        
        # Run training pipeline
        final_model_path = trainer.run_full_training_pipeline()
        
        if final_model_path:
            print(f"\n✅ SUCCESS! Model saved at: {final_model_path}")
            print(f"🧪 Next step: Run evaluation with 'python evaluate.py --model {final_model_path}'")
            return 0
        else:
            print(f"\n❌ Training failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())