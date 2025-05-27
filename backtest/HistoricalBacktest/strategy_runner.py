import backtrader as bt
import pandas as pd
import numpy as np
from CryptoTrade.strategies.TestStrategies.test_strategy import ImprovedHFT_Strategy


def create_test_data(timeframe='1min', start_date='2023-01-01', end_date='2024-01-01', base_price=50000):
    """Створення тестових даних для бектестування"""
    dates = pd.date_range(start_date, end_date, freq=timeframe)
    np.random.seed(42)

    n_bars = len(dates)

    # Генерація цін з трендом та волатільністю
    trend = np.linspace(0, 0.1, n_bars)  # Слабкий висхідний тренд
    noise = np.random.randn(n_bars).cumsum() * 0.001  # Випадкові коливання
    price_series = base_price * (1 + trend + noise)

    # OHLC дані
    opens = price_series
    closes = opens + np.random.randn(n_bars) * 10
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars) * 15)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars) * 15)
    volumes = np.random.lognormal(8, 0.5, n_bars).astype(int)

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes
    }, index=dates)

    return df


def run_hft_backtest(
        data_df=None,
        initial_cash=100000,
        commission=0.0005,
        strategy_params=None,
        print_results=True
):
    """
    Запуск HFT бектестування

    Parameters:
    -----------
    data_df : pd.DataFrame, optional
        Дані для бектестування. Якщо None, створюються тестові дані
    initial_cash : float
        Початковий капітал
    commission : float
        Комісія брокера
    strategy_params : dict, optional
        Параметри стратегії для оптимізації
    print_results : bool
        Виводити результати на екран

    Returns:
    --------
    dict: Результати бектестування
    """

    if print_results:
        print("🚀 Запуск покращеної HFT стратегії")

    # Створення даних якщо не передані
    if data_df is None:
        data_df = create_test_data()
        if print_results:
            print(f"✅ Створено {len(data_df)} хвилинних барів")

    # Налаштування бектестування
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=data_df))

    # Додавання стратегії з параметрами
    if strategy_params:
        cerebro.addstrategy(ImprovedHFT_Strategy, **strategy_params)
    else:
        cerebro.addstrategy(ImprovedHFT_Strategy)

    # Налаштування брокера
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Аналізатори
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    if print_results:
        print("⚡ Запуск бектестування...")

    # Запуск
    results = cerebro.run()

    # Отримання результатів стратегії
    strat = results[0]
    performance = strat.stop() if hasattr(strat, 'stop') else {}

    # Додаткова аналітика
    additional_metrics = {}

    # Аналіз просідання
    if hasattr(strat.analyzers, 'drawdown'):
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        if hasattr(dd_analysis, 'max') and hasattr(dd_analysis.max, 'drawdown'):
            additional_metrics['max_drawdown_pct'] = dd_analysis.max.drawdown
            additional_metrics['max_drawdown_duration'] = dd_analysis.max.len
            if print_results:
                print(f"📉 Макс. просідання: {dd_analysis.max.drawdown:.2f}%")

    # Sharpe Ratio
    if hasattr(strat.analyzers, 'sharpe'):
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        if hasattr(sharpe_analysis, 'sharperatio') and sharpe_analysis.sharperatio:
            additional_metrics['sharpe_ratio'] = sharpe_analysis.sharperatio
            if print_results:
                print(f"📊 Sharpe Ratio: {sharpe_analysis.sharperatio:.3f}")

    # Детальний аналіз угод
    if hasattr(strat.analyzers, 'trades'):
        trade_analysis = strat.analyzers.trades.get_analysis()
        if hasattr(trade_analysis, 'total') and trade_analysis.total.total > 0:
            additional_metrics['total_closed_trades'] = trade_analysis.total.total

            if hasattr(trade_analysis, 'won') and hasattr(trade_analysis, 'lost'):
                additional_metrics['avg_win'] = trade_analysis.won.pnl.average if trade_analysis.won.total > 0 else 0
                additional_metrics['avg_loss'] = trade_analysis.lost.pnl.average if trade_analysis.lost.total > 0 else 0
                additional_metrics['largest_win'] = trade_analysis.won.pnl.max if trade_analysis.won.total > 0 else 0
                additional_metrics['largest_loss'] = trade_analysis.lost.pnl.max if trade_analysis.lost.total > 0 else 0

                if print_results and trade_analysis.won.total > 0:
                    print(f"💹 Середній виграш: ${trade_analysis.won.pnl.average:.2f}")
                    print(f"📈 Найбільший виграш: ${trade_analysis.won.pnl.max:.2f}")
                if print_results and trade_analysis.lost.total > 0:
                    print(f"📉 Середній програш: ${trade_analysis.lost.pnl.average:.2f}")

    if print_results:
        print("🎯 Бектестування завершено!")

    # Повертаємо повний набір результатів
    return {
        'cerebro': cerebro,
        'results': results,
        'strategy': strat,
        'performance': performance,
        'additional_metrics': additional_metrics,
        'final_value': cerebro.broker.get_value(),
        'data_bars': len(data_df)
    }


def optimize_strategy_parameters():
    """Приклад оптимізації параметрів стратегії"""
    print("🔧 Запуск оптимізації параметрів...")

    # Параметри для тестування
    test_params = [
        {'ema_fast': 2, 'ema_slow': 6, 'position_size': 0.7},
        {'ema_fast': 3, 'ema_slow': 8, 'position_size': 0.8},
        {'ema_fast': 4, 'ema_slow': 10, 'position_size': 0.9},
    ]

    best_result = None
    best_return = -float('inf')

    for i, params in enumerate(test_params):
        print(f"\n📊 Тест #{i + 1}: {params}")
        result = run_hft_backtest(strategy_params=params, print_results=False)

        if result['performance']:
            current_return = result['performance']['total_return_pct']
            print(f"   📈 Прибутковість: {current_return:+.2f}%")

            if current_return > best_return:
                best_return = current_return
                best_result = {'params': params, 'result': result}

    if best_result:
        print(f"\n🏆 НАЙКРАЩІ ПАРАМЕТРИ:")
        print(f"   Параметри: {best_result['params']}")
        print(f"   Прибутковість: {best_return:+.2f}%")

    return best_result


def run_multiple_timeframes():
    """Тестування на різних таймфреймах"""
    timeframes = ['1min', '5min', '15min']
    results = {}

    print("📊 Тестування на різних таймфреймах...")

    for tf in timeframes:
        print(f"\n⏰ Таймфрейм: {tf}")
        data = create_test_data(timeframe=tf)
        result = run_hft_backtest(data_df=data, print_results=False)

        if result['performance']:
            results[tf] = result['performance']['total_return_pct']
            print(f"   📈 Прибутковість: {results[tf]:+.2f}%")

    # Найкращий таймфрейм
    if results:
        best_tf = max(results.keys(), key=lambda x: results[x])
        print(f"\n🏆 Найкращий таймфрейм: {best_tf} ({results[best_tf]:+.2f}%)")

    return results


if __name__ == '__main__':
    # Основне бектестування
    print("=" * 60)
    print("🚀 ОСНОВНЕ БЕКТЕСТУВАННЯ")
    print("=" * 60)

    backtest_results = run_hft_backtest()

    # Приклад використання результатів
    if backtest_results['performance']:
        performance = backtest_results['performance']
        print(f"\n📊 ШВИДКИЙ ДОСТУП ДО РЕЗУЛЬТАТІВ:")
        print(f"💰 Прибуток: ${performance['profit_loss']:+,.2f}")
        print(f"📈 Відсоток прибутку: {performance['total_return_pct']:+.2f}%")
        print(f"🏆 Win Rate: {performance['win_rate_pct']:.1f}%")

    # Додаткові тести (розкоментуйте при потребі)

    # print("\n" + "="*60)
    # print("🔧 ОПТИМІЗАЦІЯ ПАРАМЕТРІВ")
    # print("="*60)
    # optimize_strategy_parameters()

    # print("\n" + "="*60)
    # print("⏰ ТЕСТУВАННЯ ТАЙМФРЕЙМІВ")
    # print("="*60)
    # run_multiple_timeframes()