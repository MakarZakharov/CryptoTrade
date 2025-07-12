#!/usr/bin/env python3
"""
Основной скрипт для обучения DRL агентов торговли криптовалютой.
Реализует полный пайплайн от сбора данных до оценки модели.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing.data_collector import CryptoDataCollector, DataConfig
from data_processing.feature_engineering import FeatureEngineer, DataNormalizer
from environment.trading_env import create_trading_environment, TradingConfig
from agents.base_agent import AgentFactory, get_default_config
from evaluation.backtester import Backtester, BacktestConfig


class TrainingPipeline:
    """Основной класс для управления процессом обучения."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        # Создание директории для логов
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Настройка логгера
        logger = logging.getLogger('TrainingPipeline')
        logger.setLevel(logging.INFO)
        
        # Файловый handler
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Консольный handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Добавление handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Запуск полного пайплайна обучения."""
        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБУЧЕНИЯ DRL АГЕНТА")
        self.logger.info("=" * 60)
        
        try:
            # Шаг 1: Сбор данных
            data = self._collect_data()
            if data.empty:
                raise ValueError("Не удалось собрать данные")
            
            # Шаг 2: Предобработка данных
            processed_data = self._preprocess_data(data)
            
            # Шаг 3: Разделение данных
            train_data, val_data, test_data = self._split_data(processed_data)
            
            # Шаг 4: Создание среды
            train_env = self._create_environment(train_data)
            
            # Шаг 5: Создание и обучение агента
            agent = self._train_agent(train_env)
            
            # Шаг 6: Оценка агента
            evaluation_results = self._evaluate_agent(agent, test_data)
            
            # Шаг 7: Сохранение результатов
            self._save_results(agent, evaluation_results)
            
            self.logger.info("Пайплайн обучения завершен успешно!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Ошибка в пайплайне обучения: {e}")
            raise
    
    def _collect_data(self) -> pd.DataFrame:
        """Сбор исторических данных."""
        self.logger.info("Шаг 1: Сбор данных...")
        
        # Конфигурация сбора данных
        data_config = DataConfig(
            symbol=self.config['data']['symbol'],
            timeframe=self.config['data']['timeframe'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data'].get('end_date'),
            exchange=self.config['data']['exchange']
        )
        
        # Сбор данных
        collector = CryptoDataCollector(data_config)
        data = collector.collect_ohlcv_data()
        
        if not data.empty:
            self.logger.info(f"Собрано {len(data)} записей данных")
            self.logger.info(f"Период: {data.index.min()} - {data.index.max()}")
        else:
            self.logger.error("Не удалось собрать данные")
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Предобработка и генерация признаков."""
        self.logger.info("Шаг 2: Предобработка данных...")
        
        # Генерация признаков
        feature_engineer = FeatureEngineer()
        enhanced_data = feature_engineer.add_all_features(data)
        
        # Нормализация
        normalizer = DataNormalizer()
        normalized_data = normalizer.normalize_features(
            enhanced_data, 
            method=self.config['preprocessing'].get('normalization', 'minmax')
        )
        
        self.logger.info(f"Данные обработаны: {normalized_data.shape}")
        self.logger.info(f"Признаков: {len(normalized_data.columns)}")
        
        return normalized_data
    
    def _split_data(self, data: pd.DataFrame) -> tuple:
        """Разделение данных на train/val/test."""
        self.logger.info("Шаг 3: Разделение данных...")
        
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        self.logger.info(f"Train: {len(train_data)} записей")
        self.logger.info(f"Validation: {len(val_data)} записей")
        self.logger.info(f"Test: {len(test_data)} записей")
        
        return train_data, val_data, test_data
    
    def _create_environment(self, data: pd.DataFrame):
        """Создание торговой среды."""
        self.logger.info("Шаг 4: Создание торговой среды...")
        
        # Конфигурация среды
        trading_config = TradingConfig(
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee'],
            slippage=self.config['environment']['slippage'],
            lookback_window=self.config['environment']['lookback_window']
        )
        
        # Создание среды
        env = create_trading_environment(
            data, 
            trading_config, 
            self.config['environment']['reward_function']
        )
        
        self.logger.info("Торговая среда создана")
        self.logger.info(f"Пространство наблюдений: {env.observation_space}")
        self.logger.info(f"Пространство действий: {env.action_space}")
        
        return env
    
    def _train_agent(self, env):
        """Обучение агента."""
        self.logger.info("Шаг 5: Обучение агента...")
        
        # Получение конфигурации агента
        agent_type = self.config['agent']['type']
        agent_config = get_default_config(agent_type)
        agent_config.update(self.config['agent'].get('params', {}))
        
        # Создание агента
        agent = AgentFactory.create_agent(agent_type, env, agent_config)
        
        # Обучение
        training_params = self.config['training']
        self.logger.info(f"Начало обучения {agent_type} на {training_params['total_timesteps']} шагов")
        
        agent.train(
            total_timesteps=training_params['total_timesteps'],
            eval_freq=training_params.get('eval_freq', 10000),
            n_eval_episodes=training_params.get('n_eval_episodes', 5)
        )
        
        self.logger.info("Обучение завершено")
        return agent
    
    def _evaluate_agent(self, agent, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Оценка производительности агента."""
        self.logger.info("Шаг 6: Оценка агента...")
        
        # Конфигурация бэктестинга
        backtest_config = BacktestConfig(
            initial_capital=self.config['evaluation']['initial_capital'],
            commission=self.config['evaluation']['commission'],
            benchmark=self.config['evaluation']['benchmark']
        )
        
        # Запуск бэктестинга
        backtester = Backtester(backtest_config)
        results = backtester.run_backtest(agent, test_data)
        
        # Вывод результатов
        backtester.print_results()
        
        self.logger.info("Оценка завершена")
        return results
    
    def _save_results(self, agent, evaluation_results: Dict[str, Any]):
        """Сохранение результатов."""
        self.logger.info("Шаг 7: Сохранение результатов...")
        
        # Создание директории для моделей
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Имя эксперимента
        experiment_name = f"{self.config['agent']['type']}_{self.config['data']['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Сохранение модели
        model_path = os.path.join(models_dir, experiment_name)
        agent.save(model_path)
        
        # Сохранение конфигурации и результатов
        results_data = {
            'experiment_name': experiment_name,
            'config': self.config,
            'evaluation_results': evaluation_results,
            'agent_stats': agent.get_training_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = model_path + '_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.results = results_data
        
        self.logger.info(f"Результаты сохранены:")
        self.logger.info(f"  Модель: {model_path}")
        self.logger.info(f"  Результаты: {results_file}")


def create_default_config() -> Dict[str, Any]:
    """Создание конфигурации по умолчанию."""
    return {
        'data': {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': None,
            'exchange': 'binance',
            'train_ratio': 0.7,
            'val_ratio': 0.15
        },
        'preprocessing': {
            'normalization': 'minmax'
        },
        'environment': {
            'initial_balance': 10000.0,
            'transaction_fee': 0.001,
            'slippage': 0.0005,
            'lookback_window': 50,
            'reward_function': 'profit_based'
        },
        'agent': {
            'type': 'PPO',
            'params': {}
        },
        'training': {
            'total_timesteps': 100000,
            'eval_freq': 10000,
            'n_eval_episodes': 5
        },
        'evaluation': {
            'initial_capital': 10000.0,
            'commission': 0.001,
            'benchmark': 'buy_and_hold'
        }
    }


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description='Обучение DRL агента для торговли криптовалютой')
    
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации JSON')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Торговая пара')
    parser.add_argument('--timeframe', type=str, default='1h', help='Таймфрейм')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C', 'DDPG', 'DQN'], help='Алгоритм DRL')
    parser.add_argument('--timesteps', type=int, default=100000, help='Количество шагов обучения')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Начальная дата (YYYY-MM-DD)')
    parser.add_argument('--exchange', type=str, default='binance', help='Биржа')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Обновление конфигурации из аргументов командной строки
        config['data']['symbol'] = args.symbol
        config['data']['timeframe'] = args.timeframe
        config['data']['start_date'] = args.start_date
        config['data']['exchange'] = args.exchange
        config['agent']['type'] = args.algorithm
        config['training']['total_timesteps'] = args.timesteps
    
    print("🚀 Запуск обучения DRL агента для торговли криптовалютой")
    print(f"Конфигурация:")
    print(f"  Символ: {config['data']['symbol']}")
    print(f"  Таймфрейм: {config['data']['timeframe']}")
    print(f"  Алгоритм: {config['agent']['type']}")
    print(f"  Шагов обучения: {config['training']['total_timesteps']:,}")
    print("=" * 60)
    
    # Запуск пайплайна
    pipeline = TrainingPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline()
        
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"Эксперимент: {results['experiment_name']}")
        
        # Краткие результаты
        eval_results = results['evaluation_results']
        print(f"Общая доходность: {eval_results['total_return']:.2%}")
        print(f"Коэффициент Шарпа: {eval_results['sharpe_ratio']:.2f}")
        print(f"Максимальная просадка: {eval_results['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())