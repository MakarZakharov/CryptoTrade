#!/usr/bin/env python3
"""
Demo script for Professional DRL Trading System
Quick demonstration of the complete training and evaluation pipeline
"""

import sys
import time
from pathlib import Path
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"💥 Error running {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Demo of Professional DRL Trading System')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick demo with reduced timesteps')
    parser.add_argument('--training-only', action='store_true',
                        help='Run only training phase')
    parser.add_argument('--evaluation-only', action='store_true',
                        help='Run only evaluation phase (requires existing model)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for evaluation-only mode')
    
    args = parser.parse_args()
    
    print("🎯 PROFESSIONAL DRL TRADING SYSTEM DEMO")
    print("="*60)
    print("📈 BTCUSDT 15m Timeframe Training & Evaluation")
    print("🤖 PPO Algorithm with Advanced Monitoring")
    print("📊 Comprehensive Backtesting & Analysis")
    print("="*60)
    
    if args.quick:
        print("⚡ QUICK DEMO MODE - Reduced timesteps for faster execution")
    
    # Check if we're in the right directory
    if not Path("configs/ppo_config.yaml").exists():
        print("❌ Error: Please run this script from the training directory")
        print("   Expected to find: configs/ppo_config.yaml")
        return 1
    
    success_count = 0
    total_steps = 0
    
    # Training phase
    if not args.evaluation_only:
        total_steps += 1
        if args.quick:
            command = "python train.py --total-timesteps 50000 --eval-freq 10000"
            description = "Quick Training (50K timesteps)"
        else:
            command = "python train.py"
            description = "Full Training (1M timesteps)"
        
        if run_command(command, description):
            success_count += 1
        else:
            print("🚨 Training failed - stopping demo")
            return 1
    
    # Find the most recent model for evaluation
    model_path = args.model
    if not model_path and not args.training_only:
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.zip"))
            if model_files:
                # Sort by modification time, get most recent
                model_path = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
                print(f"📁 Found model for evaluation: {model_path}")
            else:
                print("❌ No model files found for evaluation")
                if not args.training_only:
                    return 1
    
    # Evaluation phase
    if not args.training_only and model_path:
        total_steps += 1
        episodes = 5 if args.quick else 10
        command = f"python evaluate.py --model \"{model_path}\" --episodes {episodes}"
        description = f"Comprehensive Evaluation ({episodes} episodes)"
        
        if run_command(command, description):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 DEMO COMPLETED")
    print(f"{'='*60}")
    print(f"✅ Successful steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🏆 All steps completed successfully!")
        
        # Show results location
        results_dir = Path("results")
        if results_dir.exists():
            reports = list(results_dir.glob("reports/*.md"))
            plots = list(results_dir.glob("plots/*.png"))
            
            if reports:
                print(f"📄 Reports generated: {len(reports)}")
                print(f"   Latest: {sorted(reports)[-1]}")
            
            if plots:
                print(f"🎨 Plots generated: {len(plots)}")
                print(f"   Location: {results_dir / 'plots'}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review the evaluation report")
        print(f"   2. Examine the generated plots")
        print(f"   3. Modify configs/ppo_config.yaml for customization")
        print(f"   4. Run full training with: python train.py")
        
        return 0
    else:
        print("❌ Some steps failed - check the logs above")
        return 1

if __name__ == "__main__":
    exit(main())