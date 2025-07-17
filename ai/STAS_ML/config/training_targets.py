from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class ModelType(Enum):
    """Типи моделей."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    LINEAR = "linear"


class TargetType(Enum):
    """Типи цільових змінних."""
    DIRECTION = "direction"
    PRICE_CHANGE = "price_change"
    VOLATILITY = "volatility"


@dataclass
class PerformanceTargets:
    """Цільові показники ефективності моделі."""
    
    # ML метрики
    min_accuracy: Optional[float] = None          # Мінімальна точність (для класифікації)
    min_f1_score: Optional[float] = None          # Мінімальний F1-score
    min_precision: Optional[float] = None         # Мінімальна точність (precision)
    min_recall: Optional[float] = None            # Мінімальний recall
    max_mse: Optional[float] = None               # Максимальна MSE (для регресії)
    min_r2_score: Optional[float] = None          # Мінімальний R² score
    
    # Торгові метрики
    min_total_return_pct: Optional[float] = None  # Мінімальна доходність %
    max_drawdown_pct: Optional[float] = None      # Максимальна просадка %
    min_sharpe_ratio: Optional[float] = None      # Мінімальний Sharpe ratio
    min_win_rate: Optional[float] = None          # Мінімальний винрейт
    min_total_trades: Optional[int] = None        # Мінімальна кількість угод
    
    # Стабільність модели
    max_overfitting_gap: Optional[float] = None   # Максимальна різниця train-val accuracy
    min_cross_val_score: Optional[float] = None   # Мінімальний CV score


@dataclass 
class TrainingTargets:
    """Основний клас цільових результатів навчання."""
    
    # Загальні цілі для всіх моделей - ОНОВЛЕНІ БАЖАНІ РЕЗУЛЬТАТИ
    general_targets: PerformanceTargets = field(default_factory=lambda: PerformanceTargets(
        # Базові ML стандарти
        min_accuracy=0.55,           # Мінімум 55% точності (краще за випадкове)
        min_f1_score=0.50,           # Мінімум 50% F1
        max_overfitting_gap=0.10,    # Не більше 10% розриву train-val
        min_cross_val_score=0.52,    # Стабільність на кросс-валідації
        
        # БАЖАНІ ТОРГОВІ РЕЗУЛЬТАТИ - ОНОВЛЕНО
        min_total_return_pct=500.0,  # 🎯 БАЖАНИЙ ЗАРОБІТОК: 500% 
        max_drawdown_pct=60.0,       # 🎯 ПРОСАДКА: <60%
        min_sharpe_ratio=0.5,        # Мінімальний Sharpe
        min_win_rate=0.50,           # 🎯 ВІНРЕЙТ: >50%
        min_total_trades=10          # Мінімум 10 угод для статистики
    ))
    
    # Цілі для конкретних моделей
    model_specific_targets: Dict[ModelType, PerformanceTargets] = field(default_factory=lambda: {
        
        # XGBoost - найкращі результати
        ModelType.XGBOOST: PerformanceTargets(
            min_accuracy=0.75,           # 75% точності
            min_f1_score=0.70,           # 70% F1
            min_precision=0.72,          # 72% precision
            min_recall=0.68,             # 68% recall
            max_overfitting_gap=0.05,    # Максимум 5% розриву
            min_cross_val_score=0.70,    # Стабільна кросс-валідація
            
            # Торгові очікування для XGBoost
            min_total_return_pct=500.0,   # Мінімум 25% річних
            max_drawdown_pct=40.0,        # Максимум 8% просадки
            min_sharpe_ratio=1.2,        # Sharpe > 1.2
            min_win_rate=0.65,           # 65% винрейт
            min_total_trades=96          # Мінімум 20 угод
        ),
        
        # Random Forest - хороші результати
        ModelType.RANDOM_FOREST: PerformanceTargets(
            min_accuracy=0.70,           # 70% точності
            min_f1_score=0.65,           # 65% F1
            min_precision=0.68,          # 68% precision
            min_recall=0.62,             # 62% recall
            max_overfitting_gap=0.08,    # Максимум 8% розриву
            min_cross_val_score=0.65,    # Кросс-валідація
            
            # Торгові очікування для Random Forest
            min_total_return_pct=15.0,   # Мінімум 15% річних
            max_drawdown_pct=12.0,       # Максимум 12% просадки
            min_sharpe_ratio=0.8,        # Sharpe > 0.8
            min_win_rate=0.58,           # 58% винрейт
            min_total_trades=15          # Мінімум 15 угод
        ),
        
        # LSTM - середні результати (складніше в налаштуванні)
        ModelType.LSTM: PerformanceTargets(
            min_accuracy=0.65,           # 65% точності
            min_f1_score=0.60,           # 60% F1
            max_overfitting_gap=0.12,    # Максимум 12% розриву (схильна до переобучення)
            min_cross_val_score=0.60,    # Кросс-валідація
            
            # Торгові очікування для LSTM
            min_total_return_pct=10.0,   # Мінімум 10% річних
            max_drawdown_pct=18.0,       # Максимум 18% просадки
            min_sharpe_ratio=0.6,        # Sharpe > 0.6
            min_win_rate=0.52,           # 52% винрейт
            min_total_trades=12          # Мінімум 12 угод
        ),
        
        # Linear - базові результати
        ModelType.LINEAR: PerformanceTargets(
            min_accuracy=0.58,           # 58% точності
            min_f1_score=0.55,           # 55% F1
            max_overfitting_gap=0.05,    # Низький розрив (стабільна)
            min_cross_val_score=0.55,    # Стабільна кросс-валідація
            
            # Торгові очікування для Linear
            min_total_return_pct=8.0,    # Мінімум 8% річних
            max_drawdown_pct=20.0,       # Максимум 20% просадки
            min_sharpe_ratio=0.4,        # Sharpe > 0.4
            min_win_rate=0.48,           # 48% винрейт
            min_total_trades=8           # Мінімум 8 угод
        )
    })
    
    # Цілі для різних типів завдань
    target_specific_goals: Dict[TargetType, PerformanceTargets] = field(default_factory=lambda: {
        
        # Прогнозування напрямку (класифікація)
        TargetType.DIRECTION: PerformanceTargets(
            min_accuracy=0.65,           # Мінімум 65% для напрямку
            min_f1_score=0.60,           # F1 для балансу
            min_precision=0.62,          # Precision важливий для торгівлі
            min_win_rate=0.55,           # 55% винрейт мінімум
            min_total_return_pct=12.0    # 12% річних мінімум
        ),
        
        # Прогнозування зміни ціни (регресія)
        TargetType.PRICE_CHANGE: PerformanceTargets(
            max_mse=0.001,               # Максимальна MSE
            min_r2_score=0.15,           # Мінімальний R²
            min_total_return_pct=8.0,    # 8% річних мінімум
            max_drawdown_pct=25.0        # Вища просадка прийнятна
        ),
        
        # Прогнозування волатильності
        TargetType.VOLATILITY: PerformanceTargets(
            max_mse=0.01,                # Максимальна MSE для волатільності
            min_r2_score=0.10,           # Мінімальний R² 
            min_sharpe_ratio=0.3,        # Низький Sharpe прийнятний
            max_drawdown_pct=30.0        # Висока просадка прийнятна
        )
    })
    
    # Прогресивні цілі (розтяжні)
    stretch_targets: PerformanceTargets = field(default_factory=lambda: PerformanceTargets(
        # Найкращі ML показники
        min_accuracy=0.85,               # 85% точності (відмінно)
        min_f1_score=0.82,               # 82% F1 (відмінно)
        min_precision=0.85,              # 85% precision
        min_recall=0.80,                 # 80% recall
        max_overfitting_gap=0.02,        # Мінімальний розрив
        min_cross_val_score=0.80,        # Дуже стабільна модель
        
        # Найкращі торгові показники  
        min_total_return_pct=50.0,       # 50% річних (відмінно)
        max_drawdown_pct=5.0,            # Максимум 5% просадки
        min_sharpe_ratio=2.0,            # Sharpe > 2.0 (відмінно)
        min_win_rate=0.80,               # 80% винрейт (відмінно)
        min_total_trades=50              # Багато статистично значущих угод
    ))


class ModelEvaluationService:
    """Сервіс для оцінки моделі відповідно до цільових показників."""
    
    def __init__(self, targets: TrainingTargets = None):
        self.targets = targets if targets else TrainingTargets()
    
    def evaluate_model(self, metrics: Dict[str, Any], 
                      model_type: ModelType, 
                      target_type: TargetType) -> Dict[str, Any]:
        """Оцінити модель відповідно до встановлених цілей."""
        
        # Отримуємо відповідні цілі
        general_targets = self.targets.general_targets
        model_targets = self.targets.model_specific_targets.get(model_type, PerformanceTargets())
        task_targets = self.targets.target_specific_goals.get(target_type, PerformanceTargets())
        stretch_targets = self.targets.stretch_targets
        
        results = {
            'meets_minimum_requirements': True,
            'meets_model_expectations': True,
            'meets_stretch_goals': True,
            'failed_requirements': [],
            'warnings': [],
            'achievements': [],
            'overall_grade': 'F',
            'score': 0.0
        }
        
        score = 0.0
        max_score = 0.0
        
        # Перевіряємо ML метрики
        score, max_score = self._check_ml_metrics(
            metrics, general_targets, model_targets, task_targets, stretch_targets, results, score, max_score
        )
        
        # Перевіряємо торгові метрики
        score, max_score = self._check_trading_metrics(
            metrics, general_targets, model_targets, task_targets, stretch_targets, results, score, max_score
        )
        
        # Розраховуємо фінальну оцінку
        if max_score > 0:
            results['score'] = score / max_score
            
            if results['score'] >= 0.9:
                results['overall_grade'] = 'A+'
            elif results['score'] >= 0.85:
                results['overall_grade'] = 'A'
            elif results['score'] >= 0.8:
                results['overall_grade'] = 'B+'
            elif results['score'] >= 0.75:
                results['overall_grade'] = 'B'
            elif results['score'] >= 0.7:
                results['overall_grade'] = 'C+'
            elif results['score'] >= 0.65:
                results['overall_grade'] = 'C'
            elif results['score'] >= 0.6:
                results['overall_grade'] = 'D'
            else:
                results['overall_grade'] = 'F'
        
        return results
    
    def _check_ml_metrics(self, metrics, general_targets, model_targets, task_targets, stretch_targets, results, score, max_score):
        """Перевірити ML метрики."""
        
        # Accuracy
        if 'test_accuracy' in metrics or 'val_accuracy' in metrics:
            accuracy = metrics.get('test_accuracy', metrics.get('val_accuracy', 0))
            max_score += 10
            
            if accuracy >= stretch_targets.min_accuracy:
                score += 10
                results['achievements'].append(f"Відмінна точність: {accuracy:.1%}")
            elif accuracy >= model_targets.min_accuracy:
                score += 8
                results['achievements'].append(f"Хороша точність: {accuracy:.1%}")
            elif accuracy >= general_targets.min_accuracy:
                score += 6
            else:
                results['meets_minimum_requirements'] = False
                results['failed_requirements'].append(f"Низька точність: {accuracy:.1%} < {general_targets.min_accuracy:.1%}")
        
        # F1 Score
        if 'test_f1' in metrics or 'val_f1' in metrics:
            f1 = metrics.get('test_f1', metrics.get('val_f1', 0))
            max_score += 8
            
            if f1 >= stretch_targets.min_f1_score:
                score += 8
                results['achievements'].append(f"Відмінний F1: {f1:.3f}")
            elif f1 >= model_targets.min_f1_score:
                score += 6
            elif f1 >= general_targets.min_f1_score:
                score += 4
            else:
                results['failed_requirements'].append(f"Низький F1: {f1:.3f} < {general_targets.min_f1_score:.3f}")
        
        return score, max_score
    
    def _check_trading_metrics(self, metrics, general_targets, model_targets, task_targets, stretch_targets, results, score, max_score):
        """Перевірити торгові метрики."""
        
        # Доходність
        if 'trading_total_return_pct' in metrics:
            returns = metrics['trading_total_return_pct']
            max_score += 15
            
            if returns >= stretch_targets.min_total_return_pct:
                score += 15
                results['achievements'].append(f"Відмінна доходність: {returns:.1f}%")
            elif returns >= model_targets.min_total_return_pct:
                score += 12
                results['achievements'].append(f"Хороша доходність: {returns:.1f}%")
            elif returns >= general_targets.min_total_return_pct:
                score += 8
            else:
                results['meets_minimum_requirements'] = False
                results['failed_requirements'].append(f"Низька доходність: {returns:.1f}% < {general_targets.min_total_return_pct}%")
        
        # Просадка
        if 'trading_max_drawdown_pct' in metrics:
            drawdown = metrics['trading_max_drawdown_pct']
            max_score += 10
            
            if drawdown <= stretch_targets.max_drawdown_pct:
                score += 10
                results['achievements'].append(f"Відмінна просадка: {drawdown:.1f}%")
            elif drawdown <= model_targets.max_drawdown_pct:
                score += 8
            elif drawdown <= general_targets.max_drawdown_pct:
                score += 6
            else:
                results['warnings'].append(f"Висока просадка: {drawdown:.1f}% > {general_targets.max_drawdown_pct}%")
        
        # Винрейт
        if 'trading_win_rate' in metrics:
            win_rate = metrics['trading_win_rate']
            max_score += 8
            
            if win_rate >= stretch_targets.min_win_rate:
                score += 8
                results['achievements'].append(f"Відмінний винрейт: {win_rate:.1%}")
            elif win_rate >= model_targets.min_win_rate:
                score += 6
            elif win_rate >= general_targets.min_win_rate:
                score += 4
            else:
                results['warnings'].append(f"Низький винрейт: {win_rate:.1%} < {general_targets.min_win_rate:.1%}")
        
        # Sharpe Ratio
        if 'trading_sharpe_ratio' in metrics:
            sharpe = metrics['trading_sharpe_ratio']
            max_score += 7
            
            if sharpe >= stretch_targets.min_sharpe_ratio:
                score += 7
                results['achievements'].append(f"Відмінний Sharpe: {sharpe:.2f}")
            elif sharpe >= model_targets.min_sharpe_ratio:
                score += 5
            elif sharpe >= general_targets.min_sharpe_ratio:
                score += 3
            else:
                results['warnings'].append(f"Низький Sharpe: {sharpe:.2f} < {general_targets.min_sharpe_ratio}")
        
        return score, max_score
    
    def print_evaluation_report(self, evaluation_results: Dict[str, Any], model_name: str = ""):
        """Вивести звіт по оцінці моделі."""
        
        print("\n" + "="*60)
        print(f"📊 ЗВІТ ПО ОЦІНЦІ МОДЕЛІ {model_name}")
        print("="*60)
        
        print(f"🎯 ЗАГАЛЬНА ОЦІНКА: {evaluation_results['overall_grade']}")
        print(f"📈 БАЛЛ: {evaluation_results['score']:.1%}")
        
        if evaluation_results['meets_minimum_requirements']:
            print("✅ Модель відповідає мінімальним вимогам")
        else:
            print("❌ Модель НЕ відповідає мінімальним вимогам")
        
        if evaluation_results['achievements']:
            print("\n🏆 ДОСЯГНЕННЯ:")
            for achievement in evaluation_results['achievements']:
                print(f"  ✅ {achievement}")
        
        if evaluation_results['failed_requirements']:
            print("\n❌ НЕ ВИКОНАНІ ВИМОГИ:")
            for failure in evaluation_results['failed_requirements']:
                print(f"  ❌ {failure}")
        
        if evaluation_results['warnings']:
            print("\n⚠️ ПОПЕРЕДЖЕННЯ:")
            for warning in evaluation_results['warnings']:
                print(f"  ⚠️ {warning}")
        
        print("\n💡 ВИСНОВКИ:")
        if evaluation_results['overall_grade'] in ['A+', 'A']:
            print("  🎉 Відмінна модель! Рекомендується для продакшену")
        elif evaluation_results['overall_grade'] in ['B+', 'B']:
            print("  👍 Хороша модель, можна використовувати")  
        elif evaluation_results['overall_grade'] in ['C+', 'C']:
            print("  🤔 Середня модель, потребує покращень")
        else:
            print("  ❌ Слабка модель, потребує серйозних покращень")
        
        print("="*60)


# Приклад використання
if __name__ == "__main__":
    # Створюємо сервіс оцінки
    evaluator = ModelEvaluationService()
    
    # Приклад метрик моделі
    example_metrics = {
        'test_accuracy': 0.78,
        'val_f1': 0.75,
        'trading_total_return_pct': 28.5,
        'trading_max_drawdown_pct': 6.2,
        'trading_win_rate': 0.68,
        'trading_sharpe_ratio': 1.45
    }
    
    # Оцінюємо модель
    results = evaluator.evaluate_model(
        example_metrics, 
        ModelType.XGBOOST, 
        TargetType.DIRECTION
    )
    
    # Виводимо звіт
    evaluator.print_evaluation_report(results, "BTCUSDT_XGBoost_Direction")