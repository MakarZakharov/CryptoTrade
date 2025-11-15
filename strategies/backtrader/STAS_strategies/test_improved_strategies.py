#!/usr/bin/env python3
"""
Тестувальник покращених стратегій для backtrader
Порівняння оригінальних та покращених версій стратегій
"""

import sys
import os
import time
import backtrader as bt
from datetime import datetime, timedelta

# Додаємо поточну директорію до шляху
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Імпорт оригінальних стратегій
from STAS_strategy import (
    TrendFollowStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    QuickScalpStrategy,
    SimpleStrategy,
    PriceActionStrategy,
    DCAStrategy,
    GridTradingStrategy
)

# Імпорт покращених стратегій
from STAS_strategy_improved import (
    ImprovedTrendFollowStrategy,
    ImprovedMomentumStrategy,
    ImprovedBreakoutStrategy,
    ImprovedQuickScalpStrategy,
    ImprovedSimpleStrategy,
    ImprovedPriceActionStrategy,
    ImprovedDCAStrategy,
    ImprovedGridTradingStrategy
)

# Словник стратегій για порівняння
STRATEGY_COMPARISON = {
    'Trend Follow': {
        'original': TrendFollowStrategy,
        'improved': ImprovedTrendFollowStrategy,
        'improvements': ['Додано ATR фільтр волатільності', 'Динамічний стоп-лосс', 'Послаблені умови входу']
    },
    'Momentum': {
        'original': MomentumStrategy,
        'improved': ImprovedMomentumStrategy,
        'improvements': ['Послаблені умови входу', 'Додано фільтр об\'єму', 'Кращий тайминг виходу']
    },
    'Breakout': {
        'original': BreakoutStrategy,
        'improved': ImprovedBreakoutStrategy,
        'improvements': ['Менша консервативність', 'Додано тейк-профіт', 'Послаблений об\'ємний фільтр']
    },
    'Quick Scalp': {
        'original': QuickScalpStrategy,
        'improved': ImprovedQuickScalpStrategy,
        'improvements': ['Послаблені умови входу', 'Реалістичні цілі', 'Додано RSI межі']
    },
    'Simple': {
        'original': SimpleStrategy,
        'improved': ImprovedSimpleStrategy,
        'improvements': ['Послаблені умови', 'Додано SMA фільтр', 'Кращі цілі прибутку']
    },
    'Price Action': {
        'original': PriceActionStrategy,
        'improved': ImprovedPriceActionStrategy,
        'improvements': ['Додані фільтри', 'Менший розмір позиції', 'Покращений риск-менеджмент']
    },
    'DCA': {
        'original': DCAStrategy,
        'improved': ImprovedDCAStrategy,
        'improvements': ['Більш активні покупки', 'Кращий фільтр тренду', 'Нижчий поріг прибутку']
    },
    'Grid Trading': {
        'original': GridTradingStrategy,
        'improved': ImprovedGridTradingStrategy,
        'improvements': ['Менші інтервали сітки', 'Більш активна торгівля', 'Додано цільовий прибуток']
    }
}


def get_test_data(symbol='BTCUSDT', period='1y', interval='15m'):
    """Завантажує тестові дані з binance CSV файлу"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    try:
        # Шлях до даних binance (відносно поточної директорії)
        data_path = "../../../data/binance/BTCUSDC/15m/2018_12_15-now.csv"
        
        print(f"📊 Завантаження даних {symbol} з binance 15m таймфрейму...")
        
        # Завантажуємо дані з CSV
        data = pd.read_csv(data_path)
        
        # Конвертуємо timestamp в datetime і встановлюємо як індекс
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Перейменовуємо колонки для відповідності backtrader формату
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Залишаємо тільки необхідні колонки для backtrader (OHLCV)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Фільтруємо дані за останній рік якщо period='1y'
        if period == '1y':
            one_year_ago = datetime.now() - timedelta(days=365)
            data = data[data.index >= one_year_ago]
        elif period == '6m':
            six_months_ago = datetime.now() - timedelta(days=180)
            data = data[data.index >= six_months_ago]
        elif period == '3m':
            three_months_ago = datetime.now() - timedelta(days=90)
            data = data[data.index >= three_months_ago]
        elif period == '1m':
            one_month_ago = datetime.now() - timedelta(days=30)
            data = data[data.index >= one_month_ago]
        
        # Сортуємо за датою
        data = data.sort_index()
        
        # Видаляємо дублікати якщо є
        data = data[~data.index.duplicated(keep='first')]
        
        # Перевіряємо чи є дані
        if data.empty:
            print("⚠️ Завантажені дані порожні, створюємо синтетичні дані...")
            return create_synthetic_data()
        
        print(f"✅ Завантажено {len(data)} барів даних BTCUSDT 15m з binance")
        print(f"📅 Період: з {data.index[0]} до {data.index[-1]}")
        
        return data
        
    except FileNotFoundError:
        print(f"❌ Файл з даними не знайдено: {data_path}")
        print("📊 Створення синтетичних тестових даних...")
        return create_synthetic_data()
        
    except Exception as e:
        print(f"❌ Помилка завантаження binance даних: {e}")
        print("📊 Створення синтетичних тестових даних...")
        return create_synthetic_data()


def create_synthetic_data(days=365):
    """Створює синтетичні тестові дані для backtrader"""
    import pandas as pd
    import numpy as np
    
    # Генеруємо дати
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Початкова ціна
    start_price = 50000
    
    # Генеруємо випадкові зміни ціни (випадкова прогулянка з трендом)
    np.random.seed(42)  # Для відтворюваності
    returns = np.random.normal(0.001, 0.02, days)  # Середній ріст 0.1% з волатільністю 2%
    
    # Створюємо ціни
    prices = [start_price]
    for i in range(1, days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1000))  # Мінімальна ціна 1000
    
    # Створюємо OHLC дані
    data = []
    for i, price in enumerate(prices):
        # Генеруємо high/low навколо закриваючої ціни
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        
        # Генеруємо об'єм
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'Open': open_price,
            'High': max(high, price, open_price),
            'Low': min(low, price, open_price),
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    print(f"✅ Створено {len(df)} барів синтетичних даних")
    return df


def run_strategy_test(strategy_class, strategy_name, initial_cash=100000, symbol='BTC-USD', verbose=False):
    """Запускає тест обраної стратегії"""
    
    try:
        # Створюємо cerebro
        cerebro = bt.Cerebro()
        
        # Завантажуємо дані
        data = get_test_data(symbol)
        if data is None:
            return None
            
        # Конвертуємо дані для backtrader
        bt_data = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(bt_data)
        
        # Додаємо стратегію
        cerebro.addstrategy(strategy_class)
        
        # Налаштування cerebro
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% комісія
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        if verbose:
            print(f"\n🚀 ТЕСТ: {strategy_name}")
            print("=" * 50)
            print(f"💰 Початковий капітал: ${initial_cash:,}")
            print("🔄 Запуск тестування...")
        
        # Запускаємо тест
        start_time = time.time()
        results = cerebro.run()
        end_time = time.time()
        
        # Отримуємо результати
        strategy_result = results[0]
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_cash - 1) * 100
        
        # Аналізатори
        sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        drawdown = strategy_result.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        trades_analysis = strategy_result.analyzers.trades.get_analysis()
        
        total_trades = trades_analysis.get('total', {}).get('total', 0)
        won_trades = trades_analysis.get('won', {}).get('total', 0)
        lost_trades = trades_analysis.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        if verbose:
            print(f"⚡ Час виконання: {end_time - start_time:.1f} сек")
            print(f"\n📊 РЕЗУЛЬТАТИ:")
            print("=" * 40)
            print(f"💰 Фінальний капітал: ${final_value:,.2f}")
            print(f"📈 Прибутковість: {total_return:+.2f}%")
            print(f"🔄 Кількість угод: {total_trades}")
            print(f"🎯 Винрейт: {win_rate:.1f}%")
            print(f"📉 Максимальна просадка: {max_dd:.2f}%")
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe_ratio or 0
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ Помилка під час тестування: {e}")
        return None


def compare_strategies():
    """Порівнює оригінальні та покращені стратегії"""
    print("\n🚀 ПОРІВНЯННЯ ОРИГІНАЛЬНИХ ТА ПОКРАЩЕНИХ СТРАТЕГІЙ")
    print("=" * 80)
    
    comparison_results = []
    
    for strategy_name, strategy_info in STRATEGY_COMPARISON.items():
        print(f"\n📊 Тестування: {strategy_name}")
        print("-" * 50)
        
        # Показуємо покращення
        print("💡 Покращення:")
        for improvement in strategy_info['improvements']:
            print(f"   • {improvement}")
        print()
        
        # Тестуємо оригінальну стратегію
        print("🔸 Оригінальна версія...")
        original_result = run_strategy_test(
            strategy_info['original'], 
            f"{strategy_name} (Original)", 
            verbose=False
        )
        
        # Тестуємо покращену стратегію
        print("🔹 Покращена версія...")
        improved_result = run_strategy_test(
            strategy_info['improved'], 
            f"{strategy_name} (Improved)", 
            verbose=False
        )
        
        # Зберігаємо результати для порівняння
        comparison_results.append({
            'name': strategy_name,
            'original': original_result,
            'improved': improved_result,
            'improvements': strategy_info['improvements']
        })
        
        # Показуємо швидке порівняння
        if original_result and improved_result:
            orig_return = original_result['total_return']
            impr_return = improved_result['total_return']
            improvement = impr_return - orig_return
            
            print(f"✅ Результат: {orig_return:+.2f}% → {impr_return:+.2f}% ({improvement:+.2f}%)")
        elif improved_result:
            print(f"✅ Результат: Оригінал не працював → {improved_result['total_return']:+.2f}%")
        else:
            print("❌ Обидві версії не працювали")
    
    # Показуємо детальну порівняльну таблицю
    print(f"\n" + "="*120)
    print("📊 ДЕТАЛЬНА ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print("="*120)
    print(f"{'Стратегія':<15} {'Оригінал %':<12} {'Трейдів':<8} {'Покращена %':<13} {'Трейдів':<8} {'Поліпшення':<12} {'Статус':<15}")
    print("-"*120)
    
    for result in comparison_results:
        name = result['name'][:14]
        
        if result['original']:
            orig_return = f"{result['original']['total_return']:+7.2f}%"
            orig_trades = f"{result['original']['total_trades']:,}"
        else:
            orig_return = "НЕ ПРАЦЮЄ"
            orig_trades = "0"
        
        if result['improved']:
            impr_return = f"{result['improved']['total_return']:+7.2f}%"
            impr_trades = f"{result['improved']['total_trades']:,}"
            
            if result['original']:
                improvement = result['improved']['total_return'] - result['original']['total_return']
                impr_str = f"{improvement:+7.2f}%"
                
                if improvement > 5:
                    status = "🎯 ВІДМІННО"
                elif improvement > 0:
                    status = "✅ КРАЩЕ"
                elif improvement > -5:
                    status = "⚠️ ТРОХИ ГІРШЕ"
                else:
                    status = "❌ ГІРШЕ"
            else:
                impr_str = "НОВИЙ ФУНКЦІОНАЛ"
                status = "🆕 ПРАЦЮЄ ТЕПЕР"
        else:
            impr_return = "НЕ ПРАЦЮЄ"
            impr_trades = "0"
            impr_str = "БЕЗ ЗМІН"
            status = "❌ НЕ ПРАЦЮЄ"
        
        print(f"{name:<15} {orig_return:<12} {orig_trades:<8} {impr_return:<13} {impr_trades:<8} {impr_str:<12} {status:<15}")
    
    print("="*120)
    
    # Показуємо топ покращення
    working_improvements = [r for r in comparison_results if r['improved'] and r['improved']['total_return'] > 0]
    if working_improvements:
        working_improvements.sort(key=lambda x: x['improved']['total_return'], reverse=True)
        best = working_improvements[0]
        
        print(f"\n🏆 НАЙКРАЩА ПОКРАЩЕНА СТРАТЕГІЯ: {best['name']}")
        print(f"📈 Прибутковість: {best['improved']['total_return']:+.2f}%")
        print(f"🔢 Кількість угод: {best['improved']['total_trades']:,}")
        print(f"🎯 Винрейт: {best['improved']['win_rate']:.1f}%")
        print(f"📉 Просадка: {best['improved']['max_drawdown']:.2f}%")
        
        print(f"\n💡 Ключові покращення:")
        for improvement in best['improvements']:
            print(f"   • {improvement}")


def select_test_mode():
    """Вибір режиму тестування"""
    print("\n📋 РЕЖИМИ ТЕСТУВАННЯ:")
    print("=" * 50)
    print("1. Порівняти всі стратегії")
    print("2. Тестувати тільки покращені версії")
    print("3. Тестувати конкретну стратегію")
    
    while True:
        choice = input("\nОберіть режим (1-3): ").strip()
        
        if choice == '1':
            return 'compare'
        elif choice == '2':
            return 'improved_only'
        elif choice == '3':
            return 'specific'
        else:
            print("❌ Невірний вибір! Оберіть 1, 2 або 3.")


def test_improved_only():
    """Тестує тільки покращені стратегії"""
    print("\n🚀 ТЕСТУВАННЯ ПОКРАЩЕНИХ СТРАТЕГІЙ")
    print("=" * 60)
    
    improved_results = []
    
    for strategy_name, strategy_info in STRATEGY_COMPARISON.items():
        print(f"\n📊 Тестування: {strategy_name} (Improved)")
        print("-" * 40)
        
        result = run_strategy_test(
            strategy_info['improved'], 
            f"{strategy_name} (Improved)", 
            verbose=True
        )
        
        if result:
            improved_results.append({
                'name': strategy_name,
                'result': result
            })
    
    # Показуємо зведену таблицю
    if improved_results:
        print(f"\n" + "="*85)
        print("📊 ЗВЕДЕНА ТАБЛИЦЯ ПОКРАЩЕНИХ СТРАТЕГІЙ")
        print("="*85)
        print(f"{'Стратегія':<20} {'Прибуток %':<12} {'Трейдів':<8} {'Винрейт %':<10} {'Просадка %':<11} {'Статус':<12}")
        print("-"*85)
        
        # Сортуємо за прибутковістю
        improved_results.sort(key=lambda x: x['result']['total_return'], reverse=True)
        
        for item in improved_results:
            name = item['name'][:19]
            result = item['result']
            
            profit_str = f"{result['total_return']:+8.2f}%"
            trades_str = f"{result['total_trades']:,}"
            winrate_str = f"{result['win_rate']:6.1f}%"
            drawdown_str = f"{result['max_drawdown']:8.2f}%"
            
            if result['total_return'] > 20:
                status = "🎯 ВІДМІННО"
            elif result['total_return'] > 0:
                status = "✅ ПРИБУТОК"
            else:
                status = "❌ ЗБИТОК"
            
            print(f"{name:<20} {profit_str:<12} {trades_str:<8} {winrate_str:<10} {drawdown_str:<11} {status:<12}")
        
        print("="*85)


def main():
    """Головна функція"""
    print("🚀 ТЕСТУВАЛЬНИК ПОКРАЩЕНИХ ТОРГОВИХ СТРАТЕГІЙ")
    print("=" * 70)
    print("💡 Функції:")
    print("   - Порівняння оригінальних та покращених стратегій")
    print("   - Аналіз ефективності покращень")
    print("   - Детальні звіти по кожній стратегії")
    
    try:
        test_mode = select_test_mode()
        
        if test_mode == 'compare':
            compare_strategies()
        elif test_mode == 'improved_only':
            test_improved_only()
        elif test_mode == 'specific':
            # Додати вибір конкретної стратегії
            print("📝 Функція вибору конкретної стратегії буде додана")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Тестування перервано користувачем.")
    except Exception as e:
        print(f"\n❌ Непередбачена помилка: {e}")


if __name__ == "__main__":
    main()