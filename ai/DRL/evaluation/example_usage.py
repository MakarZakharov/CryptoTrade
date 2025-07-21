"""Пример использования системы бектестинга DRL агентов."""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from CryptoTrade.ai.DRL.evaluation import DRLBacktester, BacktestVisualizer, run_quick_backtest
from CryptoTrade.ai.DRL.agents import PPOAgent
from CryptoTrade.ai.DRL.config import DRLConfig, TradingConfig
from CryptoTrade.ai.DRL.environments import TradingEnv
from CryptoTrade.ai.DRL.utils import DRLLogger


def example_backtest():
    """Пример полного бектестинга обученного DRL агента."""
    
    print("🚀 Пример бектестинга DRL агента")
    print("=" * 50)
    
    # Настройка логгера
    logger = DRLLogger("backtest_example")
    
    # Конфигурации
    drl_config = DRLConfig(
        agent_type="PPO",
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=100000,
        max_position_size=0.95,
        transaction_cost=0.001,
        reward_scheme="profit_based",
        lookback_window=20,
        max_episode_steps=1000
    )
    
    try:
        # Создание и загрузка агента
        print("📊 Создание агента...")
        agent = PPOAgent(drl_config, trading_config, logger)
        
        # Для примера создаем среду (в реальности агент должен быть обучен)
        env = TradingEnv(trading_config, logger=logger)
        agent.create_model(env)
        
        # Примечание: В реальном случае здесь должна быть загрузка обученной модели
        # agent.load("path/to/trained/model.zip")
        
        print("🔬 Запуск бектеста...")
        
        # Быстрый бектест
        results = run_quick_backtest(
            agent=agent,
            config=trading_config,
            deterministic=True
        )
        
        # Вывод результатов
        print("\n📈 РЕЗУЛЬТАТЫ БЕКТЕСТА:")
        print(f"  Итоговая доходность: {results['performance']['total_return_pct']:.2f}%")
        print(f"  Коэффициент Шарпа: {results['performance']['sharpe_ratio']:.2f}")
        print(f"  Максимальная просадка: {results['performance']['max_drawdown_pct']:.2f}%")
        print(f"  Всего сделок: {results['trading']['total_trades']}")
        print(f"  Винрейт: {results['trading']['win_rate']*100:.1f}%")
        
        # Создание визуализаций
        print("\n🎨 Создание визуализаций...")
        visualizer = BacktestVisualizer(logger)
        
        # Создание дашборда
        dashboard_path = visualizer.create_summary_dashboard(
            results,
            show_plot=False  # Не показываем в примере
        )
        
        print(f"📊 Дашборд сохранен: {dashboard_path}")
        
        print("\n✅ Бектест завершен успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении бектеста: {e}")
        logger.error(f"Ошибка: {e}")


def example_advanced_backtest():
    """Пример расширенного бектестинга с полной аналитикой."""
    
    print("\n🔬 Расширенный бектест с полной аналитикой")
    print("=" * 50)
    
    logger = DRLLogger("advanced_backtest")
    
    # Конфигурации (аналогичны базовому примеру)
    drl_config = DRLConfig(agent_type="PPO")
    trading_config = TradingConfig(
        symbol="BTCUSDT",
        timeframe="1h", 
        initial_balance=50000
    )
    
    try:
        # Создание агента
        agent = PPOAgent(drl_config, trading_config, logger)
        env = TradingEnv(trading_config, logger=logger)
        agent.create_model(env)
        
        # Создание расширенного бектестера
        backtester = DRLBacktester(agent, trading_config, logger)
        
        # Запуск полного бектеста
        results = backtester.run_backtest(
            deterministic=True,
            save_results=True
        )
        
        # Подробный вывод результатов
        backtester.print_results()
        
        # Создание комплексного отчета с визуализациями
        visualizer = BacktestVisualizer(logger)
        
        # Примечание: episode_data нужно получить из результатов бектеста
        # В реальной реализации эти данные сохраняются в backtester
        report_path = visualizer.create_comprehensive_report(
            results,
            show_plots=False
        )
        
        print(f"\n📊 Полный отчет создан: {report_path}")
        
        # Получение результатов для дальнейшего анализа
        final_results = backtester.get_results()
        
        print(f"\n🎯 Итоговая производительность:")
        summary = final_results['summary']
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Расширенный бектест завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        logger.error(f"Ошибка расширенного бектеста: {e}")


def example_comparison():
    """Пример сравнения нескольких агентов."""
    
    print("\n⚖️ Сравнение производительности агентов")
    print("=" * 50)
    
    logger = DRLLogger("comparison_example")
    
    # Список агентов для сравнения
    agents_config = [
        ("PPO", {"learning_rate": 3e-4}),
        ("PPO", {"learning_rate": 1e-4}),  # Разные гиперпараметры
    ]
    
    results_comparison = []
    
    trading_config = TradingConfig(symbol="BTCUSDT", timeframe="4h")
    
    for agent_type, params in agents_config:
        try:
            print(f"\n🧪 Тестирование {agent_type} с параметрами {params}...")
            
            drl_config = DRLConfig(agent_type=agent_type, **params)
            
            # Создание агента
            if agent_type == "PPO":
                agent = PPOAgent(drl_config, trading_config, logger)
            # Можно добавить другие типы агентов
            else:
                continue
            
            env = TradingEnv(trading_config, logger=logger)
            agent.create_model(env)
            
            # Быстрый бектест
            results = run_quick_backtest(agent, trading_config)
            
            # Сохраняем результаты
            agent_results = {
                'name': f"{agent_type}_{params}",
                'results': results,
                'total_return': results['performance']['total_return_pct'],
                'sharpe_ratio': results['performance']['sharpe_ratio'],
                'max_drawdown': results['performance']['max_drawdown_pct']
            }
            
            results_comparison.append(agent_results)
            
            print(f"  ✅ {agent_type}: {results['performance']['total_return_pct']:.2f}% доходность")
            
        except Exception as e:
            print(f"  ❌ Ошибка с {agent_type}: {e}")
    
    # Сравнительный анализ
    if results_comparison:
        print(f"\n🏆 ИТОГОВОЕ СРАВНЕНИЕ:")
        print("=" * 30)
        
        # Сортируем по доходности
        results_comparison.sort(key=lambda x: x['total_return'], reverse=True)
        
        for i, result in enumerate(results_comparison, 1):
            print(f"{i}. {result['name']}: {result['total_return']:.2f}% "
                  f"(Sharpe: {result['sharpe_ratio']:.2f}, DD: {result['max_drawdown']:.2f}%)")
        
        best_agent = results_comparison[0]
        print(f"\n🥇 Лучший агент: {best_agent['name']}")
    else:
        print("❌ Нет результатов для сравнения")


if __name__ == "__main__":
    print("🤖 СИСТЕМА БЕКТЕСТИНГА DRL АГЕНТОВ")
    print("=" * 50)
    
    # Базовый пример
    example_backtest()
    
    # Расширенный пример
    example_advanced_backtest()
    
    # Сравнение агентов
    example_comparison()
    
    print(f"\n🎉 Все примеры выполнены!")
    print("\n📝 ЗАМЕТКИ:")
    print("  - В реальном использовании загружайте обученные модели с agent.load()")
    print("  - Используйте реальные исторические данные для точного бектеста")
    print("  - Настройте гиперпараметры под ваши торговые стратегии")
    print("  - Всегда проверяйте результаты на out-of-sample данных")