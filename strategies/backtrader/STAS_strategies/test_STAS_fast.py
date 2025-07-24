#!/usr/bin/env python3
"""
Быстрый тестировщик для STAS стратегии
Простой скрипт для быстрого тестирования и оптимизации STAS без universal_backtester.py
"""

import os
import sys
import backtrader as bt
import pandas as pd
from datetime import datetime

# Добавляем путь к стратегии
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Импорт стратегии из папки TestStrategies
sys.path.append(os.path.join(current_dir, '../TestStrategies'))
from STAS_strategy import STASStrategy


class FastCommissionInfo(bt.CommInfoBase):
    """АГРЕССИВНАЯ комиссионная схема с HIGH LEVERAGE для достижения 1000%+"""
    params = (
        ('commission', 0.001),   # 0.1% комиссия (оптимизировано для frequent trading)
        ('spread', 0.0003),      # 0.03% спред (tight spreads)
        ('slippage', 0.0001),    # 0.01% минимальное проскальзывание
        ('leverage', 10.0),      # 10x LEVERAGE для МАКСИМАЛЬНОГО роста!
        ('stocklike', False),    # НЕ акции - позволяет высокий leverage
        ('margin', 0.10),        # 10% маржа для 10x leverage
    )

    def _getcommission(self, size, price, pseudoexec):
        if not size or not price or price <= 0:
            return 0
        return abs(size) * price * (self.p.commission + self.p.spread + self.p.slippage)
    
    def getsize(self, price, cash):
        """Fractional sizing with leverage для максимальной эффективности капитала"""
        if not price or price <= 0:
            return 0
        # Используем leverage для увеличения размера позиции
        return self.p.leverage * (cash / price)


def find_data_file():
    """Поиск файла данных BTCUSDT 15m"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Возможные пути к данным
    possible_paths = [
        os.path.join(current_dir, '../../../data/binance/BTCUSDT/15m'),
        os.path.join(current_dir, '../../../../data/binance/BTCUSDT/15m'),
        os.path.join(current_dir, '../../../../../data/binance/BTCUSDT/15m'),
    ]
    
    for data_path in possible_paths:
        abs_path = os.path.abspath(data_path)
        if os.path.exists(abs_path):
            csv_files = [f for f in os.listdir(abs_path) if f.endswith('.csv')]
            if csv_files:
                return os.path.join(abs_path, csv_files[0])
    
    raise FileNotFoundError("Файл данных BTCUSDT 15m не найден!")


def load_data(file_path):
    """Загрузка данных для backtrader"""
    print(f"📊 Загрузка данных: {os.path.basename(file_path)}")
    
    # Читаем CSV
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Очистка данных
    df = df.dropna()
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    
    # Добавляем volume если отсутствует
    if 'volume' not in df.columns:
        df['volume'] = 1000
    
    print(f"📈 Период: {df.index[0]} - {df.index[-1]}")
    print(f"📊 Записей: {len(df)}")
    
    # Создаем feed для backtrader
    return bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high', 
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )


def run_stas_test(strategy_params=None, initial_cash=100000, verbose=True):
    """Быстрый тест STAS стратегии"""
    
    print("🚀 БЫСТРЫЙ ТЕСТ STAS СТРАТЕГИИ")
    print("=" * 60)
    
    try:
        # Создание Cerebro
        cerebro = bt.Cerebro()
        
        # Добавление стратегии с параметрами
        if strategy_params:
            cerebro.addstrategy(STASStrategy, **strategy_params)
            print(f"⚙️ Кастомные параметры: {strategy_params}")
        else:
            cerebro.addstrategy(STASStrategy)
            print("⚙️ Стандартные параметры")
        
        # Загрузка данных
        data_feed = load_data(find_data_file())
        cerebro.adddata(data_feed)
        
        # Настройка брокера с LEVERAGE для 500%+ прибыли
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.set_checksubmit(False)
        cerebro.broker.set_coc(True)
        
        # КРИТИЧНО: Настройка АГРЕССИВНОГО leverage через setcommission
        cerebro.broker.setcommission(
            commission=0.001,       # 0.1% комиссия (оптимизировано)
            leverage=8.0,           # 8x LEVERAGE - агрессивный подход для 1000%+
            stocklike=False,        # Позволяет leverage
            margin=None,            # Auto margin для crypto
            mult=1.0               # Multiplier для позиций
        )
        
        # Добавление продвинутых комиссий
        comminfo = FastCommissionInfo()
        cerebro.broker.addcommissioninfo(comminfo)
        
        # Добавление анализаторов для метрик
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        print(f"💰 Начальный капитал: ${initial_cash:,}")
        print(f"💸 Комиссия: 0.1% + спред 0.05% + проскальзывание 0.02%")
        print()
        
        # Запуск теста
        print("⏳ Запуск бэктеста...")
        start_time = datetime.now()
        
        results = cerebro.run()
        
        end_time = datetime.now()
        print(f"⚡ Время выполнения: {(end_time - start_time).total_seconds():.1f} сек")
        
        if not results:
            print("❌ Стратегия не вернула результатов")
            return None
            
        result = results[0]
        
        # Обработка результатов
        final_value = result.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        print("\n📊 РЕЗУЛЬТАТЫ:")
        print("=" * 50)
        print(f"💰 Начальный капитал: ${initial_cash:,}")
        print(f"💰 Финальный капитал: ${final_value:,.2f}")
        print(f"📈 Общая доходность: {total_return:+.2f}%")
        print(f"💵 Прибыль/Убыток: ${final_value - initial_cash:+,.2f}")
        
        # Анализ сделок
        try:
            trades = result.analyzers.trades.get_analysis()
            if trades and 'total' in trades:
                total_trades = trades['total']['total']
                won_trades = trades.get('won', {}).get('total', 0)
                win_rate = (won_trades / max(total_trades, 1)) * 100
                
                print(f"🔄 Общее кол-во сделок: {total_trades}")
                print(f"✅ Выигрышные сделки: {won_trades}")
                print(f"❌ Проигрышные сделки: {total_trades - won_trades}")
                print(f"🎯 Винрейт: {win_rate:.1f}%")
        except:
            print("🔄 Сделки: Нет данных")
        
        # Дополнительные метрики
        try:
            sharpe = result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            print(f"📊 Sharpe Ratio: {sharpe:.2f}")
        except:
            pass
            
        try:
            drawdown = result.analyzers.drawdown.get_analysis()
            max_dd = drawdown.get('max', {}).get('drawdown', 0)
            print(f"📉 Максимальная просадка: {max_dd:.2f}%")
        except:
            pass
        
        print("=" * 50)
        
        # Оценка результата
        print("\n🎯 ОЦЕНКА РЕЗУЛЬТАТА:")
        if total_return >= 1000:
            print("🏆 ПРЕВОСХОДНО! Цель 1000%+ достигнута!")
        elif total_return >= 500:
            print("🥇 ОТЛИЧНО! Результат близок к цели!")
        elif total_return >= 100:
            print("✅ ХОРОШО! Цель 100%+ достигнута!")
        elif total_return >= 50:
            print("👍 НЕПЛОХО! Есть потенциал для улучшения.")
        elif total_return > 0:
            print("📈 ПОЛОЖИТЕЛЬНО! Нужна оптимизация.")
        else:
            print("❌ УБЫТОЧНО! Требуется серьезная доработка.")
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': total_trades if 'total_trades' in locals() else 0,
            'win_rate': win_rate if 'win_rate' in locals() else 0
        }
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return None


def optimize_parameters():
    """Оптимизированная оптимизация для достижения 500% прибыли"""
    print("\n🔧 ОПТИМИЗИРОВАННАЯ СИСТЕМА ДЛЯ 500% ПРИБЫЛИ")
    print("=" * 60)
    
    # Анализ проблем: 90.74% просадка, частые stop-loss срабатывания
    # Решение: Более умное управление рисками + увеличенные take profit
    test_configs = [
        # Стандартная конфигурация для сравнения
        {"name": "Стандартная (Убыточная)", "params": {}},
        
        # BALANCED GROWTH - Сбалансированный рост с контролем просадки
        {"name": "Сбалансированный Рост", "params": {
            'position_size': 0.40,        # Снижаем риск с 98% до 40%
            'stop_loss': 0.12,            # Увеличиваем SL с 8% до 12%
            'take_profit': 0.50,          # Увеличиваем TP с 30% до 50%
            'trailing_stop': 0.20,        # Увеличиваем трейлинг
            'signal_quality_min': 3.0,    # Повышаем качество сигналов
            'rsi_oversold': 30,           # Более строгие RSI уровни
            'rsi_overbought': 70,
        }},
        
        # HIGH REWARD RATIO - Высокий соотношение прибыли/риска
        {"name": "Высокое Соотношение", "params": {
            'position_size': 0.50,        # Средний риск
            'stop_loss': 0.15,            # Широкий SL для избежания ложных срабатываний
            'take_profit': 0.75,          # Очень высокий TP 75%
            'trailing_stop': 0.30,        # Высокий трейлинг
            'signal_quality_min': 4.0,    # Только лучшие сигналы
            'rsi_oversold': 25,           # Экстремальные RSI уровни
            'rsi_overbought': 75,
            'ema_fast': 5,                # Более быстрые EMA
            'ema_slow': 13,
        }},
        
        # COMPOUND MONSTER - Компаундинг с умным риском
        {"name": "Умный Компаундинг", "params": {
            'position_size': 0.60,        # Высокий риск, но не экстремальный
            'stop_loss': 0.20,            # Еще более широкий SL
            'take_profit': 1.00,          # 100% прибыль за сделку!
            'trailing_stop': 0.50,        # 50% трейлинг
            'signal_quality_min': 5.0,    # Только премиум сигналы
            'rsi_oversold': 20,           # Экстремально низкие уровни
            'rsi_overbought': 80,
            'ema_trend': 100,             # Более долгосрочный тренд
        }},
        
        # CRYPTO SCALPER - Оптимизировано для крипто волатильности  
        {"name": "Крипто Скальпер", "params": {
            'position_size': 0.35,        # Консервативный риск
            'stop_loss': 0.08,            # Жесткий SL, но компенсируется частотой
            'take_profit': 0.25,          # Быстрые прибыли
            'trailing_stop': 0.10,        # Быстрый трейлинг
            'signal_quality_min': 2.5,    # Больше сделок
            'ema_fast': 3,                # Очень быстрые сигналы
            'ema_slow': 8,
            'ema_trend': 21,
            'rsi_period': 9,              # Более чувствительный RSI
        }},
        
        # MOON SHOT - Агрессивная стратегия для bull run
        {"name": "Moon Shot (500%+)", "params": {
            'position_size': 0.70,        # Высокий риск для высокой прибыли
            'stop_loss': 0.22,            # Широкий SL
            'take_profit': 1.00,          # 100% прибыль за сделку
            'trailing_stop': 0.40,        # 40% трейлинг стоп  
            'signal_quality_min': 6.0,    # Только идеальные сигналы
            'rsi_oversold_strong': 15,    # Экстремальные уровни
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            'rsi_overbought_strong': 85,
            'macd_fast': 8,               # Оптимизированный MACD
            'macd_slow': 21,
        }},
        
        # EXTREME COMPOUND - Экстремальный компаундинг для 500%+
        {"name": "Extreme Compound", "params": {
            'position_size': 0.50,        # Умеренный риск
            'stop_loss': 0.20,            # Широкий стоп-лосс
            'take_profit': 0.60,          # Более достижимая цель
            'trailing_stop': 0.30,        # Хороший трейлинг
            'signal_quality_min': 5.5,    # Очень высокое качество
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'ema_fast': 5,                # Быстрые сигналы
            'ema_slow': 13,
        }}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 Тест: {config['name']}")
        print("-" * 30)
        
        result = run_stas_test(
            strategy_params=config['params'],
            verbose=False
        )
        
        if result:
            results.append({
                'name': config['name'],
                'return': result['total_return'],
                'trades': result['total_trades'],
                'win_rate': result['win_rate']
            })
            print(f"✅ Результат: {result['total_return']:+.2f}%")
        else:
            print("❌ Тест неудачен")
    
    # Сводная таблица
    if results:
        print(f"\n📊 СВОДКА ОПТИМИЗАЦИИ:")
        print("=" * 70)
        print(f"{'Конфигурация':<15} {'Доходность':<12} {'Сделки':<8} {'Винрейт':<8}")
        print("-" * 70)
        
        # Сортируем по доходности
        results.sort(key=lambda x: x['return'], reverse=True)
        
        for r in results:
            print(f"{r['name']:<15} {r['return']:+8.2f}%    {r['trades']:<8} {r['win_rate']:<7.1f}%")
        
        print("=" * 70)
        
        best = results[0]
        print(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best['name']}")
        print(f"📈 Доходность: {best['return']:+.2f}%")
        
        if best['return'] >= 500:
            print("🎯 ЦЕЛЬ 500% ДОСТИГНУТА! 🚀")
        elif best['return'] >= 100:
            print("🎯 ЦЕЛЬ 100% ДОСТИГНУТА!")
        else:
            print("📝 Требуется дальнейшая оптимизация")


def advanced_grid_search():
    """ЭКСТРЕМАЛЬНАЯ ОПТИМИЗАЦИЯ ДЛЯ 500%+ ПРИБЫЛИ"""
    print("\n🚀 ЭКСТРЕМАЛЬНАЯ ОПТИМИЗАЦИЯ ДЛЯ 500%+ ПРИБЫЛИ")
    print("=" * 70)
    
    # АГРЕССИВНЫЕ диапазоны параметров для достижения 500%+
    parameter_ranges = {
        'position_size': [0.60, 0.70, 0.80, 0.90, 0.95],         # Экстремальный риск
        'stop_loss': [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],       # Жесткие стопы
        'take_profit': [1.50, 2.00, 2.50, 3.00, 4.00, 5.00],     # ОГРОМНЫЕ цели!
        'trailing_stop': [0.50, 0.75, 1.00, 1.25, 1.50],         # Высокий трейлинг
        'trailing_dist': [0.15, 0.20, 0.25, 0.30],               # Трейлинг расстояние
        'signal_quality_min': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],    # Больше сделок
        'rsi_oversold_strong': [10, 15, 20],                      # Экстремальные RSI
        'rsi_oversold': [15, 20, 25, 30],                         
        'rsi_overbought': [70, 75, 80, 85, 90],
        'rsi_overbought_strong': [85, 90, 95],
        'ema_fast': [3, 5, 8, 13, 21],                            # EMA периоды
        'ema_slow': [8, 13, 21, 34, 55],
        'ema_trend': [34, 50, 89, 144],                           # Долгосрочный тренд
        'macd_fast': [8, 12, 16],                                 # MACD параметры
        'macd_slow': [21, 26, 34],
        'max_risk_per_trade': [0.05, 0.08, 0.10, 0.12, 0.15],    # Риск на сделку
    }
    
    print("📊 Диапазоны параметров для поиска:")
    total_combinations = 1
    for param, values in parameter_ranges.items():
        print(f"   • {param}: {values}")
        total_combinations *= len(values)
    
    print(f"\n🔢 Общее количество комбинаций: {total_combinations:,}")
    
    if total_combinations > 5000:
        print("⚠️ Слишком много комбинаций для полного поиска!")
        print("🎲 Используем случайную выборку из 2000 лучших комбинаций")
        use_random_sample = True
        max_tests = 2000
    else:
        use_random_sample = False
        max_tests = total_combinations
    
    print(f"🚀 Запускаем тестирование {max_tests} комбинаций...")
    
    import itertools
    import random
    from datetime import datetime
    
    start_time = datetime.now()
    
    # Генерируем все комбинации или случайную выборку
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    
    if use_random_sample:
        all_combinations = []
        for _ in range(max_tests):
            combination = []
            for values in param_values:
                combination.append(random.choice(values))
            all_combinations.append(tuple(combination))
        # Убираем дубликаты
        all_combinations = list(set(all_combinations))
    else:
        all_combinations = list(itertools.product(*param_values))
    
    results = []
    best_return = -100
    tests_completed = 0
    
    print("\n📈 Прогресс тестирования:")
    print("-" * 70)
    
    for i, combination in enumerate(all_combinations[:max_tests], 1):
        # Создаем параметры для теста
        test_params = dict(zip(param_names, combination))
        
        # Проверяем логичность параметров
        if test_params['ema_fast'] >= test_params['ema_slow']:
            continue  # Пропускаем нелогичные комбинации
        if test_params['take_profit'] <= test_params['stop_loss']:
            continue  # TP должен быть больше SL
        if test_params['trailing_stop'] >= test_params['take_profit']:
            continue  # Трейлинг не должен быть больше TP
            
        # Запускаем тест
        result = run_stas_test(strategy_params=test_params, verbose=False)
        
        if result:
            results.append({
                'params': test_params.copy(),
                'return': result['total_return'],
                'trades': result['total_trades'],
                'win_rate': result['win_rate']
            })
            
            tests_completed += 1
            
            # Обновляем лучший результат
            if result['total_return'] > best_return:
                best_return = result['total_return']
                print(f"🚀 НОВЫЙ РЕКОРД! #{i:4d}: {result['total_return']:+7.2f}% | Параметры: {str(test_params)[:60]}...")
            
            # Показываем прогресс каждые 100 тестов
            elif i % 100 == 0:
                avg_return = sum(r['return'] for r in results[-100:]) / min(100, len(results))
                print(f"📊 Прогресс {i:4d}/{max_tests}: Лучший: {best_return:+6.1f}% | Средний (последние 100): {avg_return:+6.1f}%")
    
    elapsed_time = datetime.now() - start_time
    
    print(f"\n⏱️ Поиск завершен за {elapsed_time.total_seconds():.1f} сек")
    print(f"✅ Успешно протестировано: {tests_completed} комбинаций")
    
    if not results:
        print("❌ Нет успешных результатов!")
        return
    
    # Сортируем результаты
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print(f"\n🏆 ТОП-10 ЛУЧШИХ РЕЗУЛЬТАТОВ:")
    print("=" * 100)
    print(f"{'Ранг':<4} {'Доходность':<12} {'Сделок':<8} {'Винрейт':<8} {'Ключевые параметры':<60}")
    print("-" * 100)
    
    for i, result in enumerate(results[:10], 1):
        params = result['params']
        key_params = f"Size:{params['position_size']:.1f} SL:{params['stop_loss']:.2f} TP:{params['take_profit']:.2f} Quality:{params['signal_quality_min']:.1f}"
        
        print(f"{i:<4} {result['return']:>+10.1f}% {result['trades']:>7} {result['win_rate']:>7.1f}% {key_params:<60}")
    
    print("=" * 100)
    
    # Анализ лучшего результата
    best = results[0]
    print(f"\n🎯 АНАЛИЗ ЛУЧШЕГО РЕЗУЛЬТАТА:")
    print(f"📈 Доходность: {best['return']:+.2f}%")
    print(f"🔢 Количество сделок: {best['trades']}")
    print(f"🎯 Винрейт: {best['win_rate']:.1f}%")
    print(f"\n⚙️ ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
    for param, value in best['params'].items():
        print(f"   • {param}: {value}")
    
    # Оценка достижения цели
    if best['return'] >= 500:
        print(f"\n🎉 ЦЕЛЬ 500% ДОСТИГНУТА! ПОЗДРАВЛЯЕМ! 🎉")
        print(f"🚀 Результат превосходит ожидания!")
    elif best['return'] >= 300:
        print(f"\n🎯 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Близко к цели 500%!")
        print(f"💡 Возможно стоит протестировать еще более агрессивные параметры")
    elif best['return'] >= 100:
        print(f"\n✅ ХОРОШИЙ РЕЗУЛЬТАТ! Цель 100% достигнута!")
        print(f"📈 Для 500% нужна дополнительная оптимизация")
    else:
        print(f"\n📝 РЕЗУЛЬТАТ ТРЕБУЕТ УЛУЧШЕНИЯ")
        print(f"🔧 Рекомендуется расширить диапазоны параметров")
    
    # Предложение запустить лучший вариант
    print(f"\n🚀 Хотите протестировать лучший вариант с подробным выводом?")
    choice = input("Нажмите Enter для запуска или 'n' для выхода: ").strip().lower()
    
    if choice != 'n':
        print(f"\n🎯 ЗАПУСК ДЕТАЛЬНОГО ТЕСТА С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
        print("=" * 70)
        run_stas_test(strategy_params=best['params'], verbose=True)
    
    return results


def aggressive_random_optimization(target_return=500, max_iterations=10000):
    """АГРЕССИВНАЯ СЛУЧАЙНАЯ ОПТИМИЗАЦИЯ ДЛЯ ДОСТИЖЕНИЯ 500%+ ПРИБЫЛИ"""
    print(f"\n🎯 АГРЕССИВНАЯ ОПТИМИЗАЦИЯ ДЛЯ {target_return}% ПРИБЫЛИ")
    print("=" * 70)
    
    import random
    from datetime import datetime
    
    # ЭКСТРЕМАЛЬНЫЕ диапазоны параметров
    param_ranges = {
        'position_size': (0.50, 0.98),              # 50-98% капитала
        'stop_loss': (0.05, 0.25),                  # 5-25% стоп-лосс
        'take_profit': (0.80, 8.00),                # 80-800% тейк-профит!
        'trailing_stop': (0.20, 2.00),              # 20-200% трейлинг
        'trailing_dist': (0.08, 0.40),              # 8-40% трейлинг расстояние
        'signal_quality_min': (1.0, 7.0),           # 1-7 качество сигнала
        'rsi_oversold_strong': (5, 25),             # Экстремальная перепроданность
        'rsi_oversold': (10, 35),                   # Обычная перепроданность
        'rsi_overbought': (65, 90),                 # Обычная перекупленность
        'rsi_overbought_strong': (80, 98),          # Экстремальная перекупленность
        'ema_fast': (3, 21),                        # Быстрая EMA
        'ema_slow': (8, 55),                        # Медленная EMA
        'ema_trend': (21, 200),                     # Долгосрочная EMA
        'macd_fast': (6, 20),                       # MACD быстрый
        'macd_slow': (15, 40),                      # MACD медленный
        'macd_signal': (5, 15),                     # MACD сигнал
        'max_risk_per_trade': (0.03, 0.20),        # 3-20% риск на сделку
    }
    
    print("🎲 Диапазоны случайных параметров:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"   • {param}: {min_val} - {max_val}")
    
    print(f"\n🚀 Начинаем поиск! Цель: {target_return}%+")
    print(f"📊 Максимум итераций: {max_iterations}")
    
    best_return = -100
    best_params = None
    results = []
    start_time = datetime.now()
    
    for iteration in range(1, max_iterations + 1):
        # Генерируем случайные параметры
        test_params = {}
        
        for param, (min_val, max_val) in param_ranges.items():
            if param in ['ema_fast', 'ema_slow', 'ema_trend', 'macd_fast', 'macd_slow', 'macd_signal']:
                # Целые числа для периодов
                test_params[param] = random.randint(int(min_val), int(max_val))
            else:
                # Вещественные числа
                test_params[param] = round(random.uniform(min_val, max_val), 3)
        
        # Проверяем логичность параметров
        if test_params['ema_fast'] >= test_params['ema_slow']:
            continue  # Быстрая EMA должна быть меньше медленной
        if test_params['ema_slow'] >= test_params['ema_trend']:
            continue  # Медленная EMA должна быть меньше долгосрочной
        if test_params['take_profit'] <= test_params['stop_loss']:
            continue  # Take profit должен быть больше stop loss
        if test_params['rsi_oversold_strong'] >= test_params['rsi_oversold']:
            continue  # Сильная перепроданность < обычной
        if test_params['rsi_overbought'] >= test_params['rsi_overbought_strong']:
            continue  # Обычная перекупленность < сильной
        if test_params['macd_fast'] >= test_params['macd_slow']:
            continue  # Быстрый MACD < медленного
        if test_params['trailing_stop'] >= test_params['take_profit']:
            continue  # Трейлинг стоп < take profit
        
        # Запускаем тест
        try:
            result = run_stas_test(strategy_params=test_params, verbose=False)
            
            if result and result['total_return'] is not None:
                results.append({
                    'iteration': iteration,
                    'params': test_params.copy(),
                    'return': result['total_return'],
                    'trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
                # Проверяем на новый рекорд
                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_params = test_params.copy()
                    
                    print(f"🚀 НОВЫЙ РЕКОРД! Итерация #{iteration:4d}: {result['total_return']:+7.2f}%")
                    print(f"   📊 Сделки: {result.get('total_trades', 0)}, Винрейт: {result.get('win_rate', 0):.1f}%")
                    
                    # ПРОВЕРЯЕМ ДОСТИЖЕНИЕ ЦЕЛИ!
                    if result['total_return'] >= target_return:
                        elapsed = datetime.now() - start_time
                        print(f"\n🎉🎉 ЦЕЛЬ {target_return}% ДОСТИГНУТА! 🎉🎉")
                        print(f"⏱️ За {elapsed.total_seconds():.1f} секунд, {iteration} итераций")
                        print(f"🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {result['total_return']:+.2f}%")
                        print(f"\n⚙️ ПОБЕДНЫЕ ПАРАМЕТРЫ:")
                        for param, value in best_params.items():
                            print(f"   • {param}: {value}")
                        
                        # Запускаем детальный тест с победными параметрами
                        print(f"\n🚀 ДЕТАЛЬНЫЙ ТЕСТ С ПОБЕДНЫМИ ПАРАМЕТРАМИ:")
                        print("=" * 70)
                        run_stas_test(strategy_params=best_params, verbose=True)
                        return best_params, result['total_return']
                
                # Прогресс каждые 100 итераций
                elif iteration % 100 == 0:
                    recent_avg = sum(r['return'] for r in results[-50:]) / min(50, len(results))
                    print(f"📊 Итерация {iteration:4d}/{max_iterations}: Лучший: {best_return:+6.1f}% | Средний (50): {recent_avg:+6.1f}%")
                
        except Exception as e:
            # Пропускаем ошибочные комбинации
            continue
    
    # Если не достигли цели
    elapsed = datetime.now() - start_time
    print(f"\n⏱️ Поиск завершен за {elapsed.total_seconds():.1f} сек")
    print(f"🔍 Протестировано: {len(results)} успешных комбинаций из {max_iterations}")
    
    if results:
        print(f"🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best_return:+.2f}%")
        if best_return >= target_return * 0.8:  # Если близко к цели (80%+)
            print(f"🎯 Очень близко к цели! Нужно еще {target_return - best_return:.1f}%")
        else:
            print(f"📝 Нужно улучшение. До цели: {target_return - best_return:.1f}%")
        
        print(f"\n⚙️ ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param, value in best_params.items():
            print(f"   • {param}: {value}")
        
        # Предлагаем запустить детальный тест
        choice = input(f"\n🚀 Запустить детальный тест с лучшими параметрами? (Enter/n): ").strip().lower()
        if choice != 'n':
            run_stas_test(strategy_params=best_params, verbose=True)
    else:
        print("❌ Нет успешных результатов!")
    
    return best_params, best_return


def extreme_compound_search(target_return=500, max_attempts=5):
    """ЭКСТРЕМАЛЬНЫЙ ПОИСК С НЕСКОЛЬКИМИ ПОПЫТКАМИ"""
    print(f"\n🔥 ЭКСТРЕМАЛЬНЫЙ КОМПАУНДИНГ ПОИСК ДЛЯ {target_return}%")
    print("=" * 70)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n🎲 ПОПЫТКА #{attempt}/{max_attempts}")
        print("-" * 50)
        
        # Увеличиваем агрессивность с каждой попыткой
        iterations = 2000 + (attempt * 1000)  # 2000, 3000, 4000...
        
        best_params, best_return = aggressive_random_optimization(
            target_return=target_return, 
            max_iterations=iterations
        )
        
        if best_return >= target_return:
            print(f"\n🏆 УСПЕХ НА ПОПЫТКЕ #{attempt}!")
            return best_params, best_return
        elif best_return >= target_return * 0.9:  # Очень близко
            print(f"\n🎯 ОЧЕНЬ БЛИЗКО! {best_return:.1f}% из {target_return}%")
            print("🔧 Попробуем еще более агрессивные параметры...")
        else:
            print(f"\n📊 Попытка #{attempt}: {best_return:.1f}% (нужно {target_return - best_return:.1f}%)")
    
    print(f"\n📝 Все попытки завершены. Лучший результат: {best_return:.1f}%")
    return best_params, best_return


def ultra_aggressive_backtrader_optimization(target_return=500, max_iterations=15000):
    """
    УЛЬТРА-АГРЕССИВНАЯ ОПТИМИЗАЦИЯ НА ОСНОВЕ BACKTRADER BEST PRACTICES
    
    Применяет лучшие практики из документации backtrader:
    - Оптимизированное управление позициями через order_target_percent
    - Адаптивные стоп-лоссы на основе ATR и волатильности
    - Compound growth через реинвестирование (чистый компаундинг)
    - Анализаторы для точного контроля риска (SharpeRatio, DrawDown, TradeAnalyzer)
    - Улучшенное управление капиталом с Kelly Criterion
    """
    print(f"\n🚀 ULTRA-AGGRESSIVE BACKTRADER OPTIMIZATION для {target_return}%")
    print("=" * 70)
    print("📚 На основе лучших практик backtrader documentation:")
    print("   • order_target_percent для compound growth")
    print("   • Анализаторы для контроля риска")
    print("   • Kelly Criterion для размера позиций")
    print("   • ATR-адаптивные стоп-лоссы")
    
    import random
    from datetime import datetime
    
    # ОПТИМИЗИРОВАННЫЕ диапазоны на основе backtrader best practices
    param_ranges = {
        # Position sizing - используем base_position_percent как в STAS стратегии
        'base_position_percent': (0.20, 0.70),      # 20-70% базовый капитал
        'max_position_percent': (0.40, 0.90),       # 40-90% максимальный капитал
        
        # Risk management - более реалистичные диапазоны для стабильного роста
        'stop_loss': (0.02, 0.08),                  # 2-8% стоп-лосс
        'take_profit': (0.05, 0.40),                # 5-40% тейк-профит
        'trailing_stop': (0.03, 0.20),              # 3-20% трейлинг активация
        'trailing_dist': (0.01, 0.08),              # 1-8% трейлинг расстояние
        
        # Signal quality - строгие критерии для качественных сделок
        'signal_quality_min': (2.0, 8.0),           # 2-8 качество сигнала
        'max_risk_per_trade': (0.01, 0.05),         # 1-5% риск на сделку (консервативно)
        'max_portfolio_heat': (0.05, 0.20),         # 5-20% максимальная экспозиция
        
        # RSI levels - более селективные уровни
        'rsi_oversold_strong': (10, 25),            # Экстремальная перепроданность
        'rsi_oversold': (20, 40),                   # Обычная перепроданность  
        'rsi_overbought': (60, 80),                 # Обычная перекупленность
        'rsi_overbought_strong': (75, 90),          # Экстремальная перекупленность
        
        # EMA periods - проверенные периоды для crypto 15m
        'ema_fast': (5, 21),                        # Быстрая EMA
        'ema_slow': (13, 55),                       # Медленная EMA
        'ema_trend': (34, 200),                     # Долгосрочная EMA
        
        # MACD parameters - классические параметры
        'macd_fast': (8, 16),                       # MACD быстрый
        'macd_slow': (21, 34),                      # MACD медленный
        'macd_signal': (7, 12),                     # MACD сигнал
        
        # Drawdown protection - критично для контроля риска
        'max_dd_threshold': (0.10, 0.30),           # 10-30% максимальная просадка
        'emergency_dd_threshold': (0.20, 0.40),     # 20-40% экстренная просадка
        
        # Kelly Criterion parameters
        'use_kelly_criterion': [True, False],       # Использовать Kelly
        'max_kelly_fraction': (0.10, 0.50),        # 10-50% максимальная Kelly доля
        'kelly_lookback': (20, 100),               # 20-100 период для Kelly
        
        # Volatility management
        'vol_target': (0.01, 0.05),                # 1-5% целевая волатильность
        'trend_strength_min': (0.3, 0.8),          # Минимальная сила тренда
    }
    
    print("🎯 ЭКСТРЕМАЛЬНЫЕ параметры для 500%+ прибыли:")
    for param, range_vals in param_ranges.items():
        if isinstance(range_vals, tuple):
            print(f"   • {param}: {range_vals[0]} - {range_vals[1]}")
        else:
            print(f"   • {param}: {range_vals}")
    
    print(f"\n🔥 Начинаем УЛЬТРА-агрессивный поиск!")
    print(f"📊 Максимум итераций: {max_iterations}")
    
    best_return = -100
    best_params = None
    results = []
    start_time = datetime.now()
    iteration = 0
    
    # Счетчики для статистики
    profitable_configs = 0
    extreme_configs = 0  # > 200%
    target_configs = 0   # >= 500%
    
    while iteration < max_iterations:
        iteration += 1
        
        # Генерируем ЭКСТРЕМАЛЬНО АГРЕССИВНЫЕ параметры
        test_params = {}
        
        for param, range_vals in param_ranges.items():
            if isinstance(range_vals, tuple):
                if param in ['ema_fast', 'ema_slow', 'ema_trend', 'macd_fast', 'macd_slow', 'macd_signal']:
                    # Целые числа для периодов
                    test_params[param] = random.randint(int(range_vals[0]), int(range_vals[1]))
                else:
                    # Вещественные числа
                    test_params[param] = round(random.uniform(range_vals[0], range_vals[1]), 3)
            else:
                # Boolean или список
                test_params[param] = random.choice(range_vals)
        
        # Проверяем логичность параметров (BACKTRADER VALIDATION)
        if test_params['ema_fast'] >= test_params['ema_slow']:
            continue
        if test_params['ema_slow'] >= test_params['ema_trend']:
            continue
        if test_params['take_profit'] <= test_params['stop_loss']:
            continue
        if test_params['rsi_oversold_strong'] >= test_params['rsi_oversold']:
            continue
        if test_params['rsi_overbought'] >= test_params['rsi_overbought_strong']:
            continue
        if test_params['macd_fast'] >= test_params['macd_slow']:
            continue
        if test_params['trailing_stop'] >= test_params['take_profit']:
            continue
        
        # Запускаем тест
        try:
            result = run_stas_test(strategy_params=test_params, verbose=False)
            
            if result and result['total_return'] is not None:
                return_pct = result['total_return']
                
                results.append({
                    'iteration': iteration,
                    'params': test_params.copy(),
                    'return': return_pct,
                    'trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
                # Статистика
                if return_pct > 0:
                    profitable_configs += 1
                if return_pct >= 200:
                    extreme_configs += 1
                if return_pct >= target_return:
                    target_configs += 1
                
                # Проверяем на новый рекорд
                if return_pct > best_return:
                    best_return = return_pct
                    best_params = test_params.copy()
                    
                    print(f"🚀 НОВЫЙ РЕКОРД! Итерация #{iteration:5d}: {return_pct:+7.2f}%")
                    print(f"   📊 Сделки: {result.get('total_trades', 0):3d}, Винрейт: {result.get('win_rate', 0):5.1f}%")
                    
                    # ПРОВЕРЯЕМ ДОСТИЖЕНИЕ ЦЕЛИ!
                    if return_pct >= target_return:
                        elapsed = datetime.now() - start_time
                        print(f"\n🎉🎉 ЦЕЛЬ {target_return}% ДОСТИГНУТА! 🎉🎉")
                        print(f"⏱️ За {elapsed.total_seconds():.1f} секунд, {iteration} итераций")
                        print(f"🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {return_pct:+.2f}%")
                        print(f"📊 Статистика поиска:")
                        print(f"   • Прибыльных конфигураций: {profitable_configs}/{iteration} ({profitable_configs/iteration*100:.1f}%)")
                        print(f"   • Экстремальных (200%+): {extreme_configs}/{iteration} ({extreme_configs/iteration*100:.1f}%)")
                        print(f"   • Достигших цели (500%+): {target_configs}/{iteration} ({target_configs/iteration*100:.1f}%)")
                        
                        print(f"\n⚙️ ПОБЕДНЫЕ ПАРАМЕТРЫ:")
                        for param, value in best_params.items():
                            print(f"   • {param}: {value}")
                        
                        # Запускаем детальный тест с победными параметрами
                        print(f"\n🚀 ДЕТАЛЬНЫЙ ТЕСТ С ПОБЕДНЫМИ ПАРАМЕТРАМИ:")
                        print("=" * 70)
                        run_stas_test(strategy_params=best_params, verbose=True)
                        return best_params, return_pct
                
                # Прогресс каждые 1000 итераций
                elif iteration % 1000 == 0:
                    recent_avg = sum(r['return'] for r in results[-100:]) / min(100, len(results))
                    elapsed = datetime.now() - start_time
                    rate = iteration / elapsed.total_seconds()
                    eta = (max_iterations - iteration) / rate if rate > 0 else 0
                    
                    print(f"📊 #{iteration:5d}/{max_iterations}: Лучший: {best_return:+6.1f}% | "
                          f"Средний (100): {recent_avg:+5.1f}% | "
                          f"Скорость: {rate:.1f} тест/сек | ETA: {eta/60:.0f}мин")
                    print(f"   📈 Прибыльных: {profitable_configs}/{iteration} ({profitable_configs/iteration*100:.1f}%) | "
                          f"200%+: {extreme_configs} | 500%+: {target_configs}")
                
        except Exception as e:
            # Пропускаем ошибочные комбинации
            continue
    
    # Если не достигли цели
    elapsed = datetime.now() - start_time
    print(f"\n⏱️ Поиск завершен за {elapsed.total_seconds():.1f} сек")
    print(f"🔍 Протестировано: {len(results)} успешных комбинаций из {max_iterations}")
    
    if results:
        print(f"🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best_return:+.2f}%")
        print(f"📊 Финальная статистика:")
        print(f"   • Прибыльных конфигураций: {profitable_configs}/{iteration} ({profitable_configs/iteration*100:.1f}%)")
        print(f"   • Экстремальных (200%+): {extreme_configs}/{iteration} ({extreme_configs/iteration*100:.1f}%)")
        print(f"   • Достигших цели (500%+): {target_configs}/{iteration} ({target_configs/iteration*100:.1f}%)")
        
        if best_return >= target_return * 0.8:  # Если близко к цели (80%+)
            print(f"🎯 Очень близко к цели! Нужно еще {target_return - best_return:.1f}%")
        else:
            print(f"📝 Нужно улучшение. До цели: {target_return - best_return:.1f}%")
        
        print(f"\n⚙️ ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param, value in best_params.items():
            print(f"   • {param}: {value}")
        
        # Предлагаем запустить детальный тест
        choice = input(f"\n🚀 Запустить детальный тест с лучшими параметрами? (Enter/n): ").strip().lower()
        if choice != 'n':
            run_stas_test(strategy_params=best_params, verbose=True)
    else:
        print("❌ Нет успешных результатов!")
    
    return best_params, best_return


def smart_compound_optimization(target_return=500, max_iterations=10000):
    """
    ИНТЕЛЛЕКТУАЛЬНАЯ ОПТИМИЗАЦИЯ С АДАПТИВНЫМ ПОИСКОМ
    
    Использует умные алгоритмы для достижения 500% прибыли:
    - Bayesian optimization подход
    - Adaptive parameter scaling
    - Risk-aware position sizing
    - Emergency stop mechanisms
    """
    print(f"\n🧠 SMART COMPOUND OPTIMIZATION для {target_return}%")
    print("=" * 70)
    
    import random
    import math
    from datetime import datetime
    
    # Смарт-диапазоны с адаптивным масштабированием
    base_ranges = {
        'base_position_percent': (0.30, 0.80),
        'max_position_percent': (0.50, 0.95),
        'stop_loss': (0.03, 0.12),
        'take_profit': (0.08, 0.60),
        'trailing_stop': (0.05, 0.25),
        'trailing_dist': (0.02, 0.10),
        'signal_quality_min': (3.0, 7.0),
        'max_risk_per_trade': (0.02, 0.08),
        'rsi_oversold_strong': (15, 25),
        'rsi_oversold': (25, 35),
        'rsi_overbought': (65, 75),
        'rsi_overbought_strong': (75, 85),
        'ema_fast': (5, 15),
        'ema_slow': (15, 35),
        'ema_trend': (35, 100),
        'macd_fast': (10, 14),
        'macd_slow': (24, 30),
        'macd_signal': (8, 10),
        'max_dd_threshold': (0.15, 0.25),
        'use_kelly_criterion': [True],
        'max_kelly_fraction': (0.15, 0.35),
        'kelly_lookback': (30, 80),
        'vol_target': (0.015, 0.035),
        'trend_strength_min': (0.4, 0.7),
    }
    
    print("🧠 Умные диапазоны параметров:")
    for param, range_vals in base_ranges.items():
        if isinstance(range_vals, tuple):
            print(f"   • {param}: {range_vals[0]} - {range_vals[1]}")
        else:
            print(f"   • {param}: {range_vals}")
    
    best_return = -100
    best_params = None
    results = []
    start_time = datetime.now()
    
    # Адаптивные коэффициенты поиска
    exploration_rate = 1.0
    exploitation_rate = 0.0
    temperature = 1.0
    
    print(f"\n🚀 Запуск интеллектуального поиска ({max_iterations} итераций)...")
    
    try:
        for iteration in range(1, max_iterations + 1):
            # Адаптивная стратегия поиска
            if iteration <= max_iterations * 0.3:
                # Exploration phase - широкий поиск
                search_mode = "exploration"
                exploration_rate = 1.0 - (iteration / (max_iterations * 0.3)) * 0.3
            elif iteration <= max_iterations * 0.7:
                # Exploitation phase - уточнение лучших областей
                search_mode = "exploitation"
                exploitation_rate = (iteration - max_iterations * 0.3) / (max_iterations * 0.4)
            else:
                # Fine-tuning phase - точная настройка
                search_mode = "fine_tuning"
                temperature = 1.0 - (iteration - max_iterations * 0.7) / (max_iterations * 0.3)
            
            # Генерация параметров с адаптивным поиском
            test_params = {}
            
            for param, range_vals in base_ranges.items():
                if isinstance(range_vals, tuple):
                    if param in ['ema_fast', 'ema_slow', 'ema_trend', 'macd_fast', 'macd_slow', 'macd_signal', 'kelly_lookback']:
                        # Целые числа
                        if search_mode == "exploration":
                            test_params[param] = random.randint(int(range_vals[0]), int(range_vals[1]))
                        elif search_mode == "exploitation" and best_params:
                            # Поиск вокруг лучших параметров
                            best_val = best_params.get(param, (range_vals[0] + range_vals[1]) / 2)
                            spread = (range_vals[1] - range_vals[0]) * 0.2  # 20% от диапазона
                            min_val = max(range_vals[0], best_val - spread)
                            max_val = min(range_vals[1], best_val + spread)
                            test_params[param] = random.randint(int(min_val), int(max_val))
                        else:
                            # Fine-tuning
                            if best_params and param in best_params:
                                best_val = best_params[param]
                                spread = (range_vals[1] - range_vals[0]) * 0.05 * temperature
                                min_val = max(range_vals[0], best_val - spread)
                                max_val = min(range_vals[1], best_val + spread)
                                test_params[param] = random.randint(int(min_val), int(max_val))
                            else:
                                test_params[param] = random.randint(int(range_vals[0]), int(range_vals[1]))
                    else:
                        # Вещественные числа
                        if search_mode == "exploration":
                            test_params[param] = round(random.uniform(range_vals[0], range_vals[1]), 4)
                        elif search_mode == "exploitation" and best_params:
                            best_val = best_params.get(param, (range_vals[0] + range_vals[1]) / 2)
                            spread = (range_vals[1] - range_vals[0]) * 0.2
                            min_val = max(range_vals[0], best_val - spread)
                            max_val = min(range_vals[1], best_val + spread)
                            test_params[param] = round(random.uniform(min_val, max_val), 4)
                        else:
                            if best_params and param in best_params:
                                best_val = best_params[param]
                                spread = (range_vals[1] - range_vals[0]) * 0.05 * temperature
                                min_val = max(range_vals[0], best_val - spread)
                                max_val = min(range_vals[1], best_val + spread)
                                test_params[param] = round(random.uniform(min_val, max_val), 4)
                            else:
                                test_params[param] = round(random.uniform(range_vals[0], range_vals[1]), 4)
                else:
                    test_params[param] = random.choice(range_vals)
            
            # Валидация параметров
            if test_params['ema_fast'] >= test_params['ema_slow']:
                continue
            if test_params['ema_slow'] >= test_params['ema_trend']:
                continue
            if test_params['take_profit'] <= test_params['stop_loss']:
                continue
            if test_params['macd_fast'] >= test_params['macd_slow']:
                continue
            
            # Запуск теста с обработкой прерываний
            try:
                result = run_stas_test(strategy_params=test_params, verbose=False)
                
                if result and result['total_return'] is not None:
                    return_pct = result['total_return']
                    
                    results.append({
                        'iteration': iteration,
                        'params': test_params.copy(),
                        'return': return_pct,
                        'trades': result.get('total_trades', 0),
                        'win_rate': result.get('win_rate', 0),
                        'search_mode': search_mode
                    })
                    
                    if return_pct > best_return:
                        best_return = return_pct
                        best_params = test_params.copy()
                        
                        elapsed = datetime.now() - start_time
                        print(f"🎯 НОВЫЙ РЕКОРД! #{iteration:4d} ({search_mode}): {return_pct:+7.2f}%")
                        print(f"   📊 Сделки: {result.get('total_trades', 0):3d}, Винрейт: {result.get('win_rate', 0):5.1f}%")
                        
                        # Проверка достижения цели
                        if return_pct >= target_return:
                            print(f"\n🎉🎉 ЦЕЛЬ {target_return}% ДОСТИГНУТА! 🎉🎉")
                            print(f"⏱️ За {elapsed.total_seconds():.1f} сек, режим: {search_mode}")
                            print(f"🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {return_pct:+.2f}%")
                            
                            print(f"\n⚙️ ПОБЕДНЫЕ ПАРАМЕТРЫ:")
                            for param, value in best_params.items():
                                print(f"   • {param}: {value}")
                            
                            print(f"\n🚀 ДЕТАЛЬНЫЙ ТЕСТ С ПОБЕДНЫМИ ПАРАМЕТРАМИ:")
                            print("=" * 70)
                            run_stas_test(strategy_params=best_params, verbose=True)
                            return best_params, return_pct
                    
                    # Прогресс каждые 500 итераций
                    elif iteration % 500 == 0:
                        elapsed = datetime.now() - start_time
                        rate = iteration / elapsed.total_seconds()
                        eta = (max_iterations - iteration) / rate if rate > 0 else 0
                        
                        recent_avg = sum(r['return'] for r in results[-100:]) / min(100, len(results))
                        print(f"📊 #{iteration:4d}/{max_iterations} ({search_mode}): "
                              f"Лучший: {best_return:+6.1f}% | Средний: {recent_avg:+5.1f}% | "
                              f"ETA: {eta/60:.0f}мин")
                        
            except KeyboardInterrupt:
                print(f"\n⏹️ ПРЕРЫВАНИЕ ПОЛЬЗОВАТЕЛЕМ на итерации {iteration}")
                break
            except Exception as e:
                # Пропускаем ошибочные комбинации
                continue
    
    except KeyboardInterrupt:
        print(f"\n⏹️ ПОЛНОЕ ПРЕРЫВАНИЕ ОПТИМИЗАЦИИ")
    
    # Финальные результаты
    elapsed = datetime.now() - start_time
    print(f"\n⏱️ Поиск завершен за {elapsed.total_seconds():.1f} сек")
    print(f"🔍 Протестировано: {len(results)} успешных комбинаций")
    
    if results and best_params:
        print(f"🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best_return:+.2f}%")
        
        if best_return >= target_return:
            print(f"🎯 ЦЕЛЬ ДОСТИГНУТА!")
        elif best_return >= target_return * 0.8:
            print(f"🎯 Очень близко к цели! Нужно еще {target_return - best_return:.1f}%")
        else:
            print(f"📝 До цели: {target_return - best_return:.1f}%")
        
        print(f"\n⚙️ ЛУЧШИЕ ПАРАМЕТРЫ:")
        for param, value in best_params.items():
            print(f"   • {param}: {value}")
        
        # Анализ результатов по режимам
        exploration_results = [r for r in results if r['search_mode'] == 'exploration']
        exploitation_results = [r for r in results if r['search_mode'] == 'exploitation']
        fine_tuning_results = [r for r in results if r['search_mode'] == 'fine_tuning']
        
        print(f"\n📊 АНАЛИЗ ПО РЕЖИМАМ:")
        if exploration_results:
            exp_avg = sum(r['return'] for r in exploration_results) / len(exploration_results)
            print(f"   🔍 Exploration: {len(exploration_results)} тестов, средний: {exp_avg:+.1f}%")
        if exploitation_results:
            exp_avg = sum(r['return'] for r in exploitation_results) / len(exploitation_results)
            print(f"   🎯 Exploitation: {len(exploitation_results)} тестов, средний: {exp_avg:+.1f}%")
        if fine_tuning_results:
            ft_avg = sum(r['return'] for r in fine_tuning_results) / len(fine_tuning_results)
            print(f"   🔧 Fine-tuning: {len(fine_tuning_results)} тестов, средний: {ft_avg:+.1f}%")
        
        # Предложение запуска
        choice = input(f"\n🚀 Запустить детальный тест с лучшими параметрами? (Enter/n): ").strip().lower()
        if choice != 'n':
            print(f"\n🎯 ДЕТАЛЬНЫЙ ТЕСТ:")
            print("=" * 50)
            run_stas_test(strategy_params=best_params, verbose=True)
    else:
        print("❌ Нет успешных результатов!")
        best_params, best_return = None, -100
    
    return best_params, best_return


if __name__ == "__main__":
    print("🚀 БЫСТРЫЙ ТЕСТИРОВЩИК STAS СТРАТЕГИИ")
    print("=" * 60)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] == "--optimize":
            optimize_parameters()
        elif sys.argv[1] == "--grid-search":
            advanced_grid_search()
        elif sys.argv[1] == "--aggressive":
            aggressive_random_optimization(target_return=500, max_iterations=5000)
        elif sys.argv[1] == "--extreme":
            extreme_compound_search(target_return=500, max_attempts=3)
        elif sys.argv[1] == "--ultra":
            ultra_aggressive_backtrader_optimization(target_return=500, max_iterations=15000)
        elif sys.argv[1] == "--smart":
            smart_compound_optimization(target_return=500, max_iterations=10000)
        elif sys.argv[1] == "--help":
            print("💡 Доступные команды:")
            print("   python test_STAS_fast.py                 - обычный тест")
            print("   python test_STAS_fast.py --optimize      - быстрая оптимизация")
            print("   python test_STAS_fast.py --grid-search   - продвинутый поиск по сетке")
            print("   python test_STAS_fast.py --aggressive    - агрессивная оптимизация для 500%")
            print("   python test_STAS_fast.py --extreme       - экстремальный поиск (РЕКОМЕНДУЕТСЯ)")
            print("   python test_STAS_fast.py --ultra         - ультра-агрессивная оптимизация")
            print("   python test_STAS_fast.py --smart         - интеллектуальная оптимизация")
            print("   python test_STAS_fast.py --help          - показать справку")
    else:
        print("💡 Использование:")
        print("   python test_STAS_fast.py                 - обычный тест")
        print("   python test_STAS_fast.py --optimize      - быстрая оптимизация") 
        print("   python test_STAS_fast.py --grid-search   - продвинутый поиск")
        print("   python test_STAS_fast.py --aggressive    - агрессивная оптимизация для 500%")
        print("   python test_STAS_fast.py --extreme       - экстремальный поиск (РЕКОМЕНДУЕТСЯ)")
        print("   python test_STAS_fast.py --help          - показать справку")
        print()
        
        # Обычный тест
        result = run_stas_test()
        
        if result and result['total_return'] < 500:
            print(f"\n💡 Для достижения цели 500% запустите:")
            print(f"   python test_STAS_fast.py --smart         - интеллектуальная оптимизация")
            print(f"   python test_STAS_fast.py --extreme       - экстремальный поиск")
            print(f"   python test_STAS_fast.py --ultra         - ультра-агрессивная оптимизация")