# optimized_macd_sma_optimizer.py

import backtrader as bt
import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from functools import partial
import warnings

warnings.filterwarnings('ignore')


# === Оптимизированная стратегия ===
class MACD_SMA_Strategy(bt.Strategy):
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("sma_fast", 10),
        ("sma_slow", 50),
        ("min_trades", 5),  # Минимальное количество сделок для валидности
    )

    def __init__(self):
        # Pre-calculate all indicators at once
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)

        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow)

        # Pre-calculate crossover signals
        self.macd_crossup = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.sma_trend = self.sma_fast > self.sma_slow

        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy signal: MACD crosses above signal AND fast SMA > slow SMA
            if self.macd_crossup[0] > 0 and self.sma_trend[0]:
                self.order = self.buy()
        else:
            # Sell signal: MACD crosses below signal AND fast SMA < slow SMA
            if self.macd_crossup[0] < 0 and not self.sma_trend[0]:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
            self.trade_count += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1


# === Класс для CSV данных ===
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )


# === Функция для тестирования одной комбинации параметров ===
def test_single_combination(params_combo, df_data, initial_cash=100000):
    """
    Тестирует одну комбинацию параметров
    Возвращает результаты или None если тест неуспешен
    """
    macd_fast, macd_slow, sma_fast, sma_slow = params_combo

    # Валидация параметров
    if macd_fast >= macd_slow or sma_fast >= sma_slow:
        return None

    try:
        # Создаем Cerebro для каждого теста
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)

        # Добавляем данные
        data = PandasData(dataname=df_data)
        cerebro.adddata(data)

        # Добавляем стратегию
        cerebro.addstrategy(
            MACD_SMA_Strategy,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            sma_fast=sma_fast,
            sma_slow=sma_slow
        )

        # Добавляем только необходимые анализаторы
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # Запускаем тест
        results = cerebro.run()
        strat = results[0]

        # Получаем результаты
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100

        # Анализ сделок
        trades_analysis = strat.analyzers.trades.get_analysis()
        total_trades = getattr(trades_analysis.total, 'closed', 0)

        # Пропускаем результаты с недостаточным количеством сделок
        if total_trades < 5:
            return None

        won_trades = getattr(getattr(trades_analysis, 'won', object()), 'total', 0)
        winrate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # Другие метрики
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        max_drawdown = getattr(drawdown_analysis.max, 'drawdown', 0)

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0

        # Составной скор
        composite_score = (total_return * 0.4) + (winrate * 0.3) + (float(sharpe_ratio) * 20 * 0.2) - (
                    max_drawdown * 0.1)

        return {
            'params': {
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'sma_fast': sma_fast,
                'sma_slow': sma_slow
            },
            'total_return': total_return,
            'total_trades': total_trades,
            'winrate': winrate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_value,
            'composite_score': composite_score
        }

    except Exception as e:
        return None


# === Оптимизированная функция оптимизации ===
def optimize_strategy_parallel(csv_file_path, max_workers=None):
    """
    Параллельная оптимизация параметров стратегии MACD-SMA
    """

    # Загружаем данные
    print("Загружаем данные...")
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Оптимизированные диапазоны параметров (меньше комбинаций)
    macd_fast_range = range(8, 16, 2)  # 8, 10, 12, 14
    macd_slow_range = range(20, 31, 3)  # 20, 23, 26, 29
    sma_fast_range = range(5, 21, 3)  # 5, 8, 11, 14, 17, 20
    sma_slow_range = range(30, 101, 15)  # 30, 45, 60, 75, 90

    # Генерируем все комбинации параметров
    param_combinations = list(itertools.product(
        macd_fast_range, macd_slow_range, sma_fast_range, sma_slow_range
    ))

    # Фильтруем невалидные комбинации заранее
    valid_combinations = [
        combo for combo in param_combinations
        if combo[0] < combo[1] and combo[2] < combo[3]
    ]

    total_combinations = len(valid_combinations)
    print(f"Валидных комбинаций для тестирования: {total_combinations}")

    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 8)  # Оставляем 1 ядро свободным, максимум 8

    print(f"Используем {max_workers} процессов для параллельной обработки...")

    # Создаем частичную функцию с фиксированными данными
    test_func = partial(test_single_combination, df_data=df)

    results_data = []
    completed = 0

    start_time = datetime.now()

    # Параллельная обработка
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем все задачи
        future_to_combo = {
            executor.submit(test_func, combo): combo
            for combo in valid_combinations
        }

        # Собираем результаты по мере готовности
        for future in as_completed(future_to_combo):
            completed += 1

            # Показываем прогресс каждые 10% или каждые 50 тестов
            if completed % max(1, total_combinations // 10) == 0 or completed % 50 == 0:
                elapsed = datetime.now() - start_time
                progress = completed / total_combinations * 100
                print(f"Прогресс: {completed}/{total_combinations} ({progress:.1f}%) - "
                      f"Времени прошло: {elapsed}")

            try:
                result = future.result()
                if result is not None:
                    results_data.append(result)
            except Exception as e:
                # Игнорируем ошибки отдельных тестов
                pass

    end_time = datetime.now()
    print(f"\nОптимизация завершена за: {end_time - start_time}")
    print(f"Успешно протестировано комбинаций: {len(results_data)}")

    # Сортируем по составному скору
    results_data.sort(key=lambda x: x['composite_score'], reverse=True)

    return results_data


# === Функция быстрого анализа топ результатов ===
def quick_analysis(results_data, top_n=5):
    """
    Быстрый анализ топ результатов без детального бэктестинга
    """
    print(f"\n{'=' * 80}")
    print(f"ТОП-{top_n} ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ")
    print(f"{'=' * 80}")

    for i, result in enumerate(results_data[:top_n], 1):
        params = result['params']
        print(f"\n🏆 МЕСТО #{i}")
        print(f"{'─' * 50}")
        print(f"📊 ПАРАМЕТРЫ:")
        print(f"   MACD Fast: {params['macd_fast']}")
        print(f"   MACD Slow: {params['macd_slow']}")
        print(f"   SMA Fast:  {params['sma_fast']}")
        print(f"   SMA Slow:  {params['sma_slow']}")

        print(f"\n📈 РЕЗУЛЬТАТЫ:")
        print(f"   Общий доход:     {result['total_return']:.2f}%")
        print(f"   Финальная сумма: ${result['final_value']:,.2f}")
        print(f"   Всего сделок:    {result['total_trades']}")
        print(f"   Winrate:         {result['winrate']:.2f}%")
        print(f"   Макс. просадка:  {result['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio:    {result['sharpe_ratio']:.3f}")
        print(f"   Составной скор:  {result['composite_score']:.2f}")


# === Векторизованная оптимизация (экспериментальная) ===
def vectorized_backtest(df, param_sets, initial_cash=100000):
    """
    Экспериментальная векторизованная версия бэктеста
    Может быть еще быстрее для больших объемов данных
    """
    results = []

    for params in param_sets:
        macd_fast, macd_slow, sma_fast, sma_slow = params

        if macd_fast >= macd_slow or sma_fast >= sma_slow:
            continue

        try:
            # Вычисляем индикаторы векторизованно
            close = df['close'].values

            # SMA
            sma_f = pd.Series(close).rolling(sma_fast).mean()
            sma_s = pd.Series(close).rolling(sma_slow).mean()

            # MACD (упрощенная версия)
            ema_fast = pd.Series(close).ewm(span=macd_fast).mean()
            ema_slow = pd.Series(close).ewm(span=macd_slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=9).mean()

            # Сигналы
            macd_cross_up = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
            macd_cross_down = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
            sma_uptrend = sma_f > sma_s

            # Генерируем сигналы покупки/продажи
            buy_signals = macd_cross_up & sma_uptrend
            sell_signals = macd_cross_down & ~sma_uptrend

            # Простая симуляция торговли
            position = 0
            cash = initial_cash
            shares = 0
            trades = 0

            for i in range(len(close)):
                if pd.isna(buy_signals.iloc[i]) or pd.isna(sell_signals.iloc[i]):
                    continue

                if buy_signals.iloc[i] and position == 0:
                    shares = cash / close[i]
                    cash = 0
                    position = 1
                    trades += 1

                elif sell_signals.iloc[i] and position == 1:
                    cash = shares * close[i]
                    shares = 0
                    position = 0
                    trades += 1

            # Финальная стоимость
            final_value = cash + (shares * close[-1] if shares > 0 else 0)
            total_return = (final_value - initial_cash) / initial_cash * 100

            if trades >= 5:  # Минимум сделок
                results.append({
                    'params': {'macd_fast': macd_fast, 'macd_slow': macd_slow,
                               'sma_fast': sma_fast, 'sma_slow': sma_slow},
                    'total_return': total_return,
                    'total_trades': trades,
                    'final_value': final_value,
                    'composite_score': total_return  # Упрощенный скор
                })

        except Exception:
            continue

    return sorted(results, key=lambda x: x['composite_score'], reverse=True)


# === Тестирование лучших параметров ===
def test_best_params(csv_file_path, best_params):
    """
    Тестирует лучшие параметры с подробной статистикой
    """
    print(f"\n{'=' * 80}")
    print("ПОДРОБНОЕ ТЕСТИРОВАНИЕ ЛУЧШИХ ПАРАМЕТРОВ")
    print(f"{'=' * 80}")

    # Загружаем данные
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Создаем Cerebro
    cerebro = bt.Cerebro()
    data = PandasData(dataname=df)
    cerebro.adddata(data)

    # Добавляем стратегию с лучшими параметрами
    cerebro.addstrategy(
        MACD_SMA_Strategy,
        macd_fast=best_params['macd_fast'],
        macd_slow=best_params['macd_slow'],
        sma_fast=best_params['sma_fast'],
        sma_slow=best_params['sma_slow']
    )

    # Настройки брокера
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)

    # Добавляем анализаторы
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_returns')

    # Запускаем тест
    results = cerebro.run()
    strat = results[0]

    # Выводим результаты
    starting_cash = 100000
    final_cash = cerebro.broker.getvalue()
    total_return = (final_cash - starting_cash) / starting_cash * 100

    print(f"\n📊 ИСПОЛЬЗОВАННЫЕ ПАРАМЕТРЫ:")
    print(f"   MACD Fast: {best_params['macd_fast']}")
    print(f"   MACD Slow: {best_params['macd_slow']}")
    print(f"   SMA Fast:  {best_params['sma_fast']}")
    print(f"   SMA Slow:  {best_params['sma_slow']}")

    print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал: ${starting_cash:,.2f}")
    print(f"   Финальный капитал: ${final_cash:,.2f}")
    print(f"   Общий доход:       {total_return:.2f}%")

    # Анализ сделок
    trades = strat.analyzers.trades.get_analysis()
    total_trades = getattr(trades.total, 'closed', 0)
    won_trades = getattr(getattr(trades, 'won', object()), 'total', 0)
    lost_trades = getattr(getattr(trades, 'lost', object()), 'total', 0)
    winrate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"\n📊 СТАТИСТИКА СДЕЛОК:")
    print(f"   Всего сделок:      {total_trades}")
    print(f"   Прибыльных:        {won_trades}")
    print(f"   Убыточных:         {lost_trades}")
    print(f"   Winrate:           {winrate:.2f}%")

    # Риски
    drawdown = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    print(f"\n⚠️  УПРАВЛЕНИЕ РИСКАМИ:")
    print(f"   Макс. просадка:    {drawdown.max.drawdown:.2f}%")
    print(f"   Sharpe Ratio:      {sharpe.get('sharperatio', 'N/A')}")

    # Возвращаем cerebro для возможности построения графика
    return cerebro


# === Главная функция ===
def main():
    csv_file_path = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    print("🚀 УСКОРЕННЫЙ ОПТИМИЗАТОР СТРАТЕГИИ MACD-SMA")
    print("=" * 60)

    # Выбор метода оптимизации
    print("\nВыберите метод оптимизации:")
    print("1. Параллельная оптимизация (рекомендуется)")
    print("2. Векторизованная оптимизация (экспериментальная)")

    choice = input("Введите номер (1 или 2): ").strip()

    try:
        start_time = datetime.now()

        if choice == "2":
            # Векторизованный метод
            print("\nИспользуем векторизованную оптимизацию...")
            df = pd.read_csv(csv_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Генерируем параметры
            param_sets = list(itertools.product(
                range(8, 16, 2), range(20, 31, 3),
                range(5, 21, 3), range(30, 101, 15)
            ))

            results_data = vectorized_backtest(df, param_sets)

        else:
            # Параллельный метод (по умолчанию)
            print("\nИспользуем параллельную оптимизацию...")
            results_data = optimize_strategy_parallel(csv_file_path)

        end_time = datetime.now()

        if not results_data:
            print("❌ Не найдено валидных результатов оптимизации!")
            return

        print(f"\n✅ Оптимизация завершена за: {end_time - start_time}")
        print(f"Найдено {len(results_data)} валидных комбинаций")

        # Показываем результаты
        quick_analysis(results_data, top_n=5)

        # Предлагаем детальное тестирование лучшего результата
        if input("\nВыполнить детальное тестирование лучших параметров? (y/n): ").lower().strip() == 'y':
            best_params = results_data[0]['params']
            cerebro = test_best_params(csv_file_path, best_params)

            if input("\nПостроить график? (y/n): ").lower().strip() == 'y':
                cerebro.plot()

    except FileNotFoundError:
        print(f"❌ Файл не найден: {csv_file_path}")
        print("Убедитесь, что путь к CSV файлу указан правильно.")
    except Exception as e:
        print(f"❌ Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()