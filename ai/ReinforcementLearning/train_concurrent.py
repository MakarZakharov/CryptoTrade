"""
Main entry point for concurrent distributed training
Allows multiple people to train simultaneously
"""

import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from CryptoTrade.ai.ReinforcementLearning.distributed_concurrent import ConcurrentDistributedTrainer
from CryptoTrade.ai.ReinforcementLearning.main import select_data_interactive, load_data
from CryptoTrade.ai.ReinforcementLearning.environment.trading_env import TradingEnvironment
from CryptoTrade.ai.ReinforcementLearning.agents.ppo_agent import PPOAgent
from CryptoTrade.ai.ReinforcementLearning.config import *


def show_menu():
    """Show interactive menu for training options"""
    print("\n" + "=" * 60)
    print("RL TRAINING SYSTEM - INTERACTIVE MENU")
    print("=" * 60)
    
    # Training mode selection
    print("\n1. Режим обучения:")
    print("   [1] Первый участник (инициализация нового обучения)")
    print("   [2] Другой участник (присоединиться к существующему обучению)")
    print("   [3] Простое обучение (без распределенной системы)")
    
    while True:
        try:
            mode = int(input("\nВыберите режим (1-3): "))
            if mode in [1, 2, 3]:
                break
            print("Неверный выбор. Пожалуйста, введите 1, 2 или 3.")
        except ValueError:
            print("Пожалуйста, введите число.")
    
    # Multi-threading selection
    use_multithreading = False
    num_threads = 1
    
    print("\n2. Многопоточность:")
    print("   [1] Обычное обучение (один поток)")
    print("   [2] Многопоточное обучение")
    
    while True:
        try:
            mt_choice = int(input("\nВыберите вариант (1-2): "))
            if mt_choice in [1, 2]:
                if mt_choice == 2:
                    use_multithreading = True
                    num_threads = int(input("Введите количество потоков (2-8): "))
                    num_threads = max(2, min(8, num_threads))  # Ограничение от 2 до 8
                break
            print("Неверный выбор. Пожалуйста, введите 1 или 2.")
        except ValueError:
            print("Пожалуйста, введите число.")
    
    return mode, use_multithreading, num_threads


def run_simple_training(args):
    """Run simple training without distributed system"""
    print("\n" + "=" * 60)
    print("ПРОСТОЕ ОБУЧЕНИЕ (БЕЗ РАСПРЕДЕЛЕННОЙ СИСТЕМЫ)")
    print("=" * 60)
    
    # Import simple training function
    from CryptoTrade.ai.ReinforcementLearning.main import train_rl_agent
    
    # Run training
    train_rl_agent(
        data_path=args.data,
        timeframe=args.timeframe,
        agent_type='ppo',
        episodes=args.episodes,
        validation_split=0.2,
        interactive=True
    )


def run_multithreaded_training(trainer, agent, train_env, val_env, episodes, state, num_threads):
    """Run training with multiple threads (CPU-based parallelism)"""
    import concurrent.futures
    import copy
    
    print(f"\nЗапуск параллельного обучения с {num_threads} потоками...")
    
    # For CUDA multi-threading, we need to use sequential training or multiprocessing
    # Since CUDA doesn't support true multi-threading, we'll train sequentially
    # but split the episodes
    
    if torch.cuda.is_available():
        print("ВНИМАНИЕ: CUDA не поддерживает многопоточность. Используется последовательное обучение.")
        print("Для истинной параллельности используйте CPU или запустите несколько отдельных процессов.")
        
        # Train sequentially but show progress for each "thread"
        results = []
        episodes_per_thread = episodes // num_threads
        remaining_episodes = episodes % num_threads
        
        for i in range(num_threads):
            thread_episodes = episodes_per_thread
            if i < remaining_episodes:
                thread_episodes += 1
            
            print(f"\nВыполнение части {i + 1}/{num_threads}: {thread_episodes} эпизодов")
            
            try:
                # Train this portion
                metrics = trainer.train_concurrent(
                    agent=agent,
                    train_env=train_env,
                    val_env=val_env,
                    episodes=thread_episodes,
                    state=state
                )
                results.append(metrics)
                print(f"Часть {i + 1}/{num_threads}: Завершено")
            except Exception as e:
                print(f"Ошибка в части {i + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    else:
        # CPU-based training can use threads
        def train_worker(thread_id, episodes_per_thread):
            """Worker function for each thread"""
            print(f"Поток {thread_id}: Начало обучения {episodes_per_thread} эпизодов")
            
            try:
                # Create thread-local copies of environments
                local_train_env = copy.deepcopy(train_env)
                local_val_env = copy.deepcopy(val_env)
                
                # Each thread trains a portion of episodes
                metrics = trainer.train_concurrent(
                    agent=agent,
                    train_env=local_train_env,
                    val_env=local_val_env,
                    episodes=episodes_per_thread,
                    state=state
                )
                
                print(f"Поток {thread_id}: Завершено")
                return metrics
            except Exception as e:
                print(f"Ошибка в потоке {thread_id}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Split episodes among threads
        episodes_per_thread = episodes // num_threads
        remaining_episodes = episodes % num_threads
        
        # Create futures for each thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads):
                thread_episodes = episodes_per_thread
                if i < remaining_episodes:
                    thread_episodes += 1
                
                future = executor.submit(train_worker, i + 1, thread_episodes)
                futures.append(future)
            
            # Wait for all threads to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Ошибка при получении результата: {e}")
        
        return results


def main():
    # Show interactive menu
    mode, use_multithreading, num_threads = show_menu()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Concurrent Distributed RL Training')
    parser.add_argument('--data', help='Path to data file (interactive selection if not provided)')
    parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes to train')
    parser.add_argument('--total-episodes', type=int, default=1000, help='Total target episodes')
    parser.add_argument('--shared-dir', default='concurrent_models', help='Shared models directory')
    parser.add_argument('--sync-interval', type=int, default=10, help='Episodes between syncs')
    parser.add_argument('--participant-id', help='Unique participant ID (auto-generated if not provided)')
    parser.add_argument('--no-sync', action='store_true', help='Disable automatic synchronization')
    
    args = parser.parse_args()
    
    # Handle simple training mode
    if mode == 3:
        run_simple_training(args)
        return
    
    # Set participant ID for first participant
    if mode == 1:
        args.participant_id = "first_participant"
        print("\nИнициализация как первый участник...")
    else:
        if not args.participant_id:
            args.participant_id = input("\nВведите ID участника (или оставьте пустым для автогенерации): ").strip()
        print(f"\nПрисоединение как участник: {args.participant_id or 'auto-generated'}")
    
    # Initialize concurrent trainer
    trainer = ConcurrentDistributedTrainer(
        shared_dir=args.shared_dir,
        participant_id=args.participant_id,
        sync_interval=args.sync_interval
    )
    
    print("=" * 60)
    print("CONCURRENT DISTRIBUTED RL TRAINING")
    print("=" * 60)
    print(f"Participant: {trainer.participant_id}")
    print(f"Shared directory: {args.shared_dir}")
    print(f"Sync interval: {args.sync_interval} episodes")
    print("-" * 60)
    
    # Get current training summary
    summary = trainer.get_training_summary()
    if summary:
        print("\nCurrent Training Status:")
        print(f"  Total episodes completed: {summary['total_episodes']}/{summary['target_episodes']}")
        print(f"  Active participants: {summary['active_participants']}")
        print(f"  Total participants: {summary['total_participants']}")
        print(f"  Best reward: {summary['best_reward']:.4f}")
        print(f"  Model versions: {summary['model_versions']}")
        print("-" * 60)
    
    # Select data
    if args.data:
        data_path = args.data
    else:
        data_path = select_data_interactive()
    
    # Initialize or resume training
    state = trainer.initialize_or_resume(
        data_path=data_path,
        timeframe=args.timeframe,
        total_episodes=args.total_episodes
    )
    
    # Check if training is complete
    if state['total_episodes_completed'] >= state['total_episodes_target']:
        print(f"\nTraining already complete! ({state['total_episodes_completed']}/{state['total_episodes_target']} episodes)")
        print("To continue, increase --total-episodes")
        return
    
    # Load data
    print("\nLoading data...")
    data = load_data(data_path)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create environments
    train_env = TradingEnvironment(
        data=train_data,
        timeframe=args.timeframe,
        initial_balance=INITIAL_BALANCE,
        trading_fees=TRADING_FEES,
        max_position_size=MAX_POSITION_SIZE
    )
    
    val_env = TradingEnvironment(
        data=val_data,
        timeframe=args.timeframe,
        initial_balance=INITIAL_BALANCE,
        trading_fees=TRADING_FEES,
        max_position_size=MAX_POSITION_SIZE
    )
    
    # Create or load agent
    agent = PPOAgent(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space
    )
    
    # Try to load best model if exists
    best_model_path = os.path.join(args.shared_dir, "models", "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from previous training...")
        agent.load(best_model_path)
    else:
        # Try to load latest model from any participant
        models_dir = os.path.join(args.shared_dir, "models")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and f != 'best_model.pth']
            if model_files:
                # Sort by modification time and get latest
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                latest_model = os.path.join(models_dir, model_files[0])
                print(f"\nLoading latest model: {model_files[0]}")
                agent.load(latest_model)
    
    # Calculate episodes to train
    remaining_episodes = state['total_episodes_target'] - state['total_episodes_completed']
    episodes_to_train = min(args.episodes, remaining_episodes)
    
    print(f"\nTraining {episodes_to_train} episodes...")
    print(f"Global progress: {state['total_episodes_completed']}/{state['total_episodes_target']}")
    print("-" * 60)
    
    # Train
    try:
        if use_multithreading and mode != 3:
            # Multi-threaded training
            results = run_multithreaded_training(
                trainer, agent, train_env, val_env, 
                episodes_to_train, state, num_threads
            )
            
            # Aggregate results
            if results:
                total_episodes = sum(r.get('local_episodes', 0) for r in results if r)
                best_reward = max((r.get('best_reward', -np.inf) for r in results if r), default=-np.inf)
            else:
                total_episodes = 0
                best_reward = -np.inf
            
            print("\n" + "=" * 60)
            print("MULTI-THREADED TRAINING SESSION COMPLETE")
            print("=" * 60)
            print(f"Participant: {trainer.participant_id}")
            print(f"Total episodes trained: {total_episodes}")
            print(f"Best reward: {best_reward:.4f}")
            print(f"Threads used: {num_threads}")
        else:
            # Single-threaded training
            metrics = trainer.train_concurrent(
                agent=agent,
                train_env=train_env,
                val_env=val_env,
                episodes=episodes_to_train,
                state=state
            )
            
            print("\n" + "=" * 60)
            print("TRAINING SESSION COMPLETE")
            print("=" * 60)
            print(f"Participant: {trainer.participant_id}")
            print(f"Episodes trained: {metrics['local_episodes']}")
            print(f"Global episodes: {metrics['global_episodes']}/{state['total_episodes_target']}")
            print(f"Best reward: {metrics['best_reward']:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Progress has been saved and can be resumed")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()