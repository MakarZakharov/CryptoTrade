"""
Основной тренер для ML моделей STAS_ML.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.STAS_ML.config.ml_config import MLConfig
from CryptoTrade.ai.STAS_ML.data.data_processor import CryptoDataProcessor
from CryptoTrade.ai.STAS_ML.models.predictor import CryptoPricePredictor
from CryptoTrade.ai.STAS_ML.evaluation.evaluator import ModelEvaluator


class MLTrainer:
    """Главный класс для обучения ML моделей."""
    
    def __init__(self, config: MLConfig, save_dir: str = "ml_models", custom_model_name: str = None):
        self.config = config
        self.save_dir = save_dir
        
        # Создаем имя эксперимента
        if custom_model_name:
            self.experiment_name = custom_model_name
        else:
            self.experiment_name = f"{config.symbol}_{config.timeframe}_{config.model_type}_{config.target_type}"
        
        # Создаем директории
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"logs/ml/{self.experiment_name}", exist_ok=True)
        
        # Инициализируем компоненты
        self.data_processor = CryptoDataProcessor(config)
        self.predictor = CryptoPricePredictor(config)
        self.evaluator = ModelEvaluator(config)
        
        # Данные
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Подготовить данные для обучения."""
        print("🔄 Подготавливаем данные...")
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.data_processor.prepare_data()
        
        # Сохраняем информацию о данных
        data_info = {
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'n_features': self.X_train.shape[1],
            'target_type': self.config.target_type,
            'prepared_at': datetime.now().isoformat()
        }
        
        with open(f"logs/ml/{self.experiment_name}/data_info.json", 'w') as f:
            json.dump(data_info, f, indent=2)
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def train(self) -> Dict[str, Any]:
        """Обучить модель."""
        print(f"🚀 Начинаем обучение ML модели")
        print(f"Эксперимент: {self.experiment_name}")
        print(f"Модель: {self.config.model_type}")
        print(f"Символ: {self.config.symbol}")
        print(f"Таймфрейм: {self.config.timeframe}")
        print(f"Цель: {self.config.target_type}")
        print("-" * 50)
        
        # Подготавливаем данные если еще не подготовлены
        if self.X_train is None:
            self.prepare_data()
        
        # Обучаем модель
        training_metrics = self.predictor.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        
        # Оцениваем на тестовых данных
        print("\n🔍 Оценка на тестовых данных...")
        test_predictions = self.predictor.predict(self.X_test)
        test_metrics = self.evaluator.evaluate(self.y_test, test_predictions)
        
        # Добавляем торговую симуляцию
        print("💰 Торговая симуляция...")
        trading_sim = self.evaluator.create_trading_simulation(self.y_test, test_predictions)
        
        # Рассчитываем win rate для классификации
        win_rate = 0.0
        if self.config.target_type == 'direction':
            correct_predictions = np.sum(self.y_test == test_predictions)
            total_predictions = len(self.y_test)
            win_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Объединяем все метрики
        all_metrics = {
            **training_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            'experiment_name': self.experiment_name,
            'model_type': self.config.model_type,
            'target_type': self.config.target_type,
            'trained_at': datetime.now().isoformat()
        }
        
        # Добавляем торговые метрики если симуляция прошла успешно
        if 'error' not in trading_sim:
            all_metrics.update({
                'trading_total_return_pct': trading_sim.get('total_return_pct', 0.0),
                'trading_max_drawdown_pct': trading_sim.get('max_drawdown_pct', 0.0),
                'trading_total_trades': trading_sim.get('total_trades', 0),
                'trading_win_rate': win_rate,
                'trading_sharpe_ratio': trading_sim.get('sharpe_ratio', 0.0),
                'trading_final_balance': trading_sim.get('final_balance', 0.0),
                'trading_initial_balance': trading_sim.get('initial_balance', 10000.0)
            })
        
        # Сохраняем метрики
        self.save_metrics(all_metrics)
        
        # Создаем отчет
        self.create_report(all_metrics, test_predictions)
        
        print(f"\n✅ Обучение завершено!")
        print(f"📊 Логи: logs/ml/{self.experiment_name}/")
        
        return all_metrics
    
    def save_model(self) -> str:
        """Сохранить обученную модель."""
        if self.predictor.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train().")
        
        model_path = f"{self.save_dir}/{self.experiment_name}_model.joblib"
        self.predictor.save(model_path)
        
        print(f"✅ Модель сохранена: {model_path}")
        return model_path
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Сохранить метрики."""
        # JSON для программного доступа
        with open(f"logs/ml/{self.experiment_name}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # CSV для анализа
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"logs/ml/{self.experiment_name}/metrics.csv", index=False)
    
    def create_report(self, metrics: Dict[str, Any], test_predictions: np.ndarray):
        """Создать отчет по обучению."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"ОТЧЕТ ПО ОБУЧЕНИЮ ML МОДЕЛИ")
        report_lines.append("=" * 60)
        report_lines.append(f"Эксперимент: {self.experiment_name}")
        report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Конфигурация
        report_lines.append("КОНФИГУРАЦИЯ:")
        report_lines.append(f"  Символ: {self.config.symbol}")
        report_lines.append(f"  Таймфрейм: {self.config.timeframe}")
        report_lines.append(f"  Модель: {self.config.model_type}")
        report_lines.append(f"  Цель: {self.config.target_type}")
        report_lines.append(f"  Lookback window: {self.config.lookback_window}")
        report_lines.append("")
        
        # Данные
        report_lines.append("ДАННЫЕ:")
        report_lines.append(f"  Train samples: {len(self.X_train):,}")
        report_lines.append(f"  Validation samples: {len(self.X_val):,}")
        report_lines.append(f"  Test samples: {len(self.X_test):,}")
        report_lines.append(f"  Features: {self.X_train.shape[1]:,}")
        report_lines.append("")
        
        # Метрики
        report_lines.append("МЕТРИКИ:")
        if self.config.target_type == 'direction':
            # Классификация
            report_lines.append(f"  Train Accuracy: {metrics.get('train_accuracy', 0):.4f}")
            report_lines.append(f"  Val Accuracy: {metrics.get('val_accuracy', 0):.4f}")
            report_lines.append(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            if 'test_f1' in metrics:
                report_lines.append(f"  Test F1-score: {metrics['test_f1']:.4f}")
        else:
            # Регрессия
            report_lines.append(f"  Train MSE: {metrics.get('train_mse', 0):.6f}")
            report_lines.append(f"  Val MSE: {metrics.get('val_mse', 0):.6f}")
            report_lines.append(f"  Test MSE: {metrics.get('test_mse', 0):.6f}")
            report_lines.append(f"  Train MAE: {metrics.get('train_mae', 0):.6f}")
            report_lines.append(f"  Val MAE: {metrics.get('val_mae', 0):.6f}")
            report_lines.append(f"  Test MAE: {metrics.get('test_mae', 0):.6f}")
            if 'test_r2' in metrics:
                report_lines.append(f"  Test R²: {metrics['test_r2']:.4f}")
        report_lines.append("")
        
        # Торговые метрики
        if 'trading_total_return_pct' in metrics:
            report_lines.append("ТОРГОВЫЕ РЕЗУЛЬТАТЫ:")
            report_lines.append(f"  Начальный баланс: ${metrics['trading_initial_balance']:,.2f}")
            report_lines.append(f"  Финальный баланс: ${metrics['trading_final_balance']:,.2f}")
            report_lines.append(f"  Общая доходность: {metrics['trading_total_return_pct']:+.2f}%")
            report_lines.append(f"  Максимальная просадка: {metrics['trading_max_drawdown_pct']:.2f}%")
            report_lines.append(f"  Количество сделок: {metrics['trading_total_trades']}")
            report_lines.append(f"  Процент выигрышных: {metrics['trading_win_rate']*100:.1f}%")
            if 'trading_sharpe_ratio' in metrics:
                report_lines.append(f"  Sharpe Ratio: {metrics['trading_sharpe_ratio']:.4f}")
            report_lines.append("")

        # Важность признаков (если доступна)
        feature_importance = self.predictor.get_feature_importance()
        if feature_importance is not None:
            report_lines.append("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature_idx, importance) in enumerate(sorted_features[:10]):
                if hasattr(self.data_processor, 'feature_names') and len(self.data_processor.feature_names) > feature_idx:
                    feature_name = self.data_processor.feature_names[feature_idx]
                else:
                    feature_name = f"feature_{feature_idx}"
                report_lines.append(f"  {i+1}. {feature_name}: {importance:.4f}")
            report_lines.append("")
        
        # Сохраняем отчет
        report_text = "\n".join(report_lines)
        with open(f"logs/ml/{self.experiment_name}/report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Выводим краткий отчет в консоль
        print("\n📋 КРАТКИЙ ОТЧЕТ:")
        print("-" * 50)
        if self.config.target_type == 'direction':
            print(f"Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        else:
            print(f"Test MSE: {metrics.get('test_mse', 0):.6f}")
            print(f"Test MAE: {metrics.get('test_mae', 0):.6f}")
        
        # Торговые метрики
        if 'trading_total_return_pct' in metrics:
            print(f"\n💰 ТОРГОВЫЕ РЕЗУЛЬТАТЫ:")
            print(f"   Доходность: {metrics['trading_total_return_pct']:+.2f}%")
            print(f"   Максимальная просадка: {metrics['trading_max_drawdown_pct']:.2f}%")
            print(f"   Количество сделок: {metrics['trading_total_trades']}")
            print(f"   Процент выигрышных: {metrics['trading_win_rate']*100:.1f}%")
            print(f"   Финальный баланс: ${metrics['trading_final_balance']:,.2f}")
        print("-" * 50)
    
    def cross_validate(self) -> Dict[str, Any]:
        """Выполнить кросс-валидацию."""
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        
        print(f"🔄 Выполняем {self.config.cross_validation_folds}-fold кросс-валидацию...")
        
        # Подготавливаем данные если еще не подготовлены
        if self.X_train is None:
            self.prepare_data()
        
        # Объединяем train и validation для кросс-валидации
        X_cv = np.vstack([self.X_train, self.X_val])
        y_cv = np.hstack([self.y_train, self.y_val])
        
        # Создаем модель для кросс-валидации
        cv_predictor = CryptoPricePredictor(self.config)
        cv_predictor._create_model()
        
        # Выбираем тип кросс-валидации
        if self.config.target_type == 'direction':
            cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=self.config.random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=self.config.random_state)
            scoring = 'neg_mean_squared_error'
        
        # Выполняем кросс-валидацию
        cv_scores = cross_val_score(cv_predictor.model, X_cv, y_cv, cv=cv, scoring=scoring)
        
        # Результаты
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'scoring': scoring
        }
        
        print(f"✅ Кросс-валидация завершена:")
        print(f"   Среднее: {cv_results['cv_mean']:.4f}")
        print(f"   Стандартное отклонение: {cv_results['cv_std']:.4f}")
        
        # Сохраняем результаты
        with open(f"logs/ml/{self.experiment_name}/cross_validation.json", 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return cv_results
    
    def save_config(self):
        """Сохранить конфигурацию эксперимента."""
        config_dict = {
            'exchange': self.config.exchange,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'model_type': self.config.model_type,
            'target_type': self.config.target_type,
            'prediction_horizon': self.config.prediction_horizon,
            'lookback_window': self.config.lookback_window,
            'train_split': self.config.train_split,
            'validation_split': self.config.validation_split,
            'test_split': self.config.test_split,
            'include_technical_indicators': self.config.include_technical_indicators,
            'indicator_periods': self.config.indicator_periods,
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat()
        }
        
        # JSON
        with open(f"logs/ml/{self.experiment_name}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # CSV для табличного анализа
        config_df = pd.DataFrame([config_dict])
        config_df.to_csv(f"logs/ml/{self.experiment_name}/config.csv", index=False)


def quick_train_ml(symbol: str = "BTCUSDT", timeframe: str = "1d", 
                   model_type: str = "xgboost", target_type: str = "direction",
                   custom_name: str = None) -> MLTrainer:
    """Быстрое обучение ML модели с минимальными настройками."""
    
    config = MLConfig(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        target_type=target_type
    )
    
    trainer = MLTrainer(config, custom_model_name=custom_name)
    trainer.save_config()
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение ML модели для торговли криптовалютами')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--model', default='xgboost', 
                       choices=['xgboost', 'random_forest', 'lstm', 'linear'], 
                       help='Тип модели')
    parser.add_argument('--target', default='direction',
                       choices=['direction', 'price_change', 'volatility'],
                       help='Целевая переменная')
    parser.add_argument('--name', help='Пользовательское имя модели')
    
    args = parser.parse_args()
    
    quick_train_ml(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_type=args.model,
        target_type=args.target,
        custom_name=args.name
    )