#!/usr/bin/env python3
"""
Тестувальник стратегій для backtrader
Дозволяє запускати та тестувати різні торгові стратегії
"""

import sys
import os
import time
import backtrader as bt
import yfinance as yf
from datetime import datetime, timedelta

# Додаємо поточну директорію до шляху
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Імпорт всіх стратегій
from STAS_strategy import (
    TrendFollowStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    QuickScalpStrategy,
    RSIBounceStrategy,
    VolumeSpreadStrategy,
    PriceActionStrategy,
    SimpleStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    StochasticStrategy,
    MeanReversionStrategy,
    DCAStrategy,
    GridTradingStrategy
)

# Словник доступних стратегій
AVAILABLE_STRATEGIES = {
    '1': {'class': TrendFollowStrategy, 'name': 'Trend Follow', 'description': 'Проста трендова стратегія: купуємо вище EMA + зростання'},
    '2': {'class': MomentumStrategy, 'name': 'Momentum', 'description': 'Швидка моментум стратегія: купуємо при сильному зростанні'},
    '3': {'class': BreakoutStrategy, 'name': 'Breakout', 'description': 'Пробій екстремумів: купуємо при пробої максимуму'},
    '4': {'class': QuickScalpStrategy, 'name': 'Quick Scalp', 'description': 'Швидкий скальпінг: купуємо при швидкому зростанні'},
    '5': {'class': RSIBounceStrategy, 'name': 'RSI Bounce', 'description': 'RSI відбиття: купуємо при RSI<40 та зростанні'},
    '6': {'class': VolumeSpreadStrategy, 'name': 'Volume Spread', 'description': 'Об\'ємний спред: купуємо при високому об\'ємі'},
    '7': {'class': PriceActionStrategy, 'name': 'Price Action', 'description': 'Прайс екшн: купуємо при бичачих свічках'},
    '8': {'class': SimpleStrategy, 'name': 'Simple Buy/Sell', 'description': 'Найпростіша: купуємо при зростанні ціни'},
    '9': {'class': MACDStrategy, 'name': 'MACD Crossover', 'description': 'MACD стратегія: купуємо при перетині MACD вгору'},
    '10': {'class': BollingerBandsStrategy, 'name': 'Bollinger Bands', 'description': 'Bollinger Bands: купуємо при відбитті від нижньої смуги'},
    '11': {'class': StochasticStrategy, 'name': 'Stochastic', 'description': 'Стохастична стратегія: купуємо при виході з перепроданості'},
    '12': {'class': MeanReversionStrategy, 'name': 'Mean Reversion', 'description': 'Повернення до середнього: купуємо при відхиленні від SMA'},
    '13': {'class': DCAStrategy, 'name': 'Dollar Cost Averaging', 'description': 'DCA стратегія: регулярне усереднення покупок'},
    '14': {'class': GridTradingStrategy, 'name': 'Grid Trading', 'description': 'Грід-трейдінг: створює сітку ордерів для торгівлі'},
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


def run_strategy_test(strategy_class, strategy_params=None, initial_cash=100000, symbol='BTC-USD', verbose=True):
    """Запускає тест обраної стратегії"""
    
    if verbose:
        strategy_name = strategy_class.__name__
        print(f"\n🚀 ТЕСТ СТРАТЕГІЇ: {strategy_name}")
        print("=" * 60)
    
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
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
            if verbose:
                print(f"⚙️ Параметри: {strategy_params}")
        else:
            cerebro.addstrategy(strategy_class)
            if verbose:
                print("⚙️ Стандартні параметри")
        
        # Налаштування cerebro
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% комісія
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        if verbose:
            print(f"💰 Початковий капітал: ${initial_cash:,}")
            print(f"📈 Тестові дані: {symbol}")
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
            print("=" * 50)
            print(f"💰 Початковий капітал: ${initial_cash:,}")
            print(f"💰 Фінальний капітал: ${final_value:,.2f}")
            print(f"📈 Загальна прибутковість: {total_return:+.2f}%")
            print(f"💵 Прибуток/Збиток: ${final_value - initial_cash:+,.2f}")
            print(f"🔄 Загальна кількість угод: {total_trades}")
            print(f"✅ Прибуткові угоди: {won_trades}")
            print(f"❌ Збиткові угоди: {lost_trades}")
            print(f"🎯 Винрейт: {win_rate:.1f}%")
            print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio else "📊 Sharpe Ratio: N/A")
            print(f"📉 Максимальна просадка: {max_dd:.2f}%")
            print("=" * 50)
            
            # Оцінка результату
            if total_return > 20:
                print("🎯 ВІДМІННИЙ РЕЗУЛЬТАТ! 🎉")
            elif total_return > 0:
                print("✅ Прибутковий результат")
            elif total_return > -10:
                print("⚠️ Незначний збиток")
            else:
                print("❌ Збитковий результат")
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe_ratio or 0
        }
        
    except Exception as e:
        print(f"❌ Помилка під час тестування: {e}")
        return None


def select_strategy():
    """Інтерактивний вибір стратегії"""
    print("\n📋 ДОСТУПНІ СТРАТЕГІЇ:")
    print("=" * 70)
    
    for key, strategy_info in AVAILABLE_STRATEGIES.items():
        print(f"{key}. {strategy_info['name']}")
        print(f"   💡 {strategy_info['description']}")
        print()
    
    while True:
        choice = input("Оберіть стратегію (1-14) або 'all' для всіх: ").strip().lower()
        
        if choice == 'all':
            return 'all', "Всі стратегії"
        
        if choice in AVAILABLE_STRATEGIES:
            strategy_info = AVAILABLE_STRATEGIES[choice]
            return strategy_info['class'], strategy_info['name']
        
        print("❌ Невірний вибір! Оберіть число від 1 до 14 або 'all'.")


def test_all_strategies():
    """Тестує всі стратегії та показує порівняльну таблицю"""
    print("\n🚀 ТЕСТУВАННЯ ВСІХ СТРАТЕГІЙ")
    print("=" * 60)
    
    all_results = []
    
    for key, strategy_info in AVAILABLE_STRATEGIES.items():
        strategy_class = strategy_info['class']
        strategy_name = strategy_info['name']
        
        print(f"\n📊 Тестування: {strategy_name}")
        print("-" * 40)
        
        result = run_strategy_test(strategy_class=strategy_class, verbose=False)
        
        if result:
            all_results.append({
                'name': strategy_name,
                'return': result['total_return'],
                'trades': result['total_trades'],
                'win_rate': result['win_rate'],
                'max_drawdown': result['max_drawdown'],
                'final_value': result['final_value']
            })
            print(f"✅ Завершено: {result['total_return']:+.2f}%")
        else:
            all_results.append({
                'name': strategy_name,
                'return': None,
                'trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'final_value': 100000
            })
            print("❌ Помилка тестування")
    
    # Показуємо зведену таблицю результатів
    print(f"\n" + "="*95)
    print("📊 ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ ВСІХ СТРАТЕГІЙ")
    print("="*95)
    print(f"{'Стратегія':<25} {'Прибуток %':<12} {'Трейдів':<8} {'Винрейт %':<10} {'Просадка %':<11} {'Капітал $':<12}")
    print("-"*95)
    
    # Сортуємо за прибутковістю
    all_results.sort(key=lambda x: x['return'] if x['return'] is not None else -1000, reverse=True)
    
    for result in all_results:
        name = result['name']
        if len(name) > 24:
            name = name[:21] + "..."
        
        if result['return'] is not None:
            profit_str = f"{result['return']:+8.2f}%"
            trades_str = f"{result['trades']:,}"
            winrate_str = f"{result['win_rate']:6.1f}%"
            drawdown_str = f"{result['max_drawdown']:8.2f}%"
            capital_str = f"${result['final_value']:,.0f}"
        else:
            profit_str = "ПОМИЛКА"
            trades_str = "0"
            winrate_str = "0.0%"
            drawdown_str = "0.00%"
            capital_str = "$100,000"
        
        print(f"{name:<25} {profit_str:<12} {trades_str:<8} {winrate_str:<10} {drawdown_str:<11} {capital_str:<12}")
    
    print("="*95)
    
    # Показуємо найкращу стратегію
    best = all_results[0]
    if best['return'] is not None and best['return'] > 0:
        print(f"\n🏆 НАЙКРАЩА СТРАТЕГІЯ: {best['name']}")
        print(f"📈 Прибутковість: {best['return']:+.2f}%")
        print(f"🔢 Кількість угод: {best['trades']:,}")
        print(f"🎯 Винрейт: {best['win_rate']:.1f}%")
    else:
        print(f"\n📝 Всі стратегії показали негативні результати.")
        print(f"💡 Спробуйте змінити параметри або тестові дані.")


def main():
    """Головна функція"""
    print("🚀 ТЕСТУВАЛЬНИК ТОРГОВИХ СТРАТЕГІЙ")
    print("=" * 60)
    print("💡 Використання:")
    print("   - Оберіть стратегію для тестування")
    print("   - Або протестуйте всі стратегії одночасно")
    print("   - Перегляньте результати та порівняйте ефективність")
    
    try:
        # Вибір стратегії
        strategy_class, strategy_name = select_strategy()
        
        if strategy_class == 'all':
            # Тестуємо всі стратегії
            test_all_strategies()
        else:
            # Тестуємо одну стратегію
            print(f"\n🎯 Обрана стратегія: {strategy_name}")
            result = run_strategy_test(strategy_class=strategy_class)
            
            if result:
                print(f"\n✅ Тестування завершено успішно!")
            else:
                print(f"\n❌ Сталася помилка під час тестування.")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Тестування перервано користувачем.")
    except Exception as e:
        print(f"\n❌ Непередбачена помилка: {e}")


if __name__ == "__main__":
    main()