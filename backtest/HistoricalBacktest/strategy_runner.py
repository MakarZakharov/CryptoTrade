import os
import sys
import backtrader as bt
import pandas as pd
import warnings

# Додаємо шлях до стратегії
sys.path.append(os.path.join(os.path.dirname(__file__), '../../strategies/TestStrategies'))
from test_strategy import ProfitableBTCStrategy

warnings.filterwarnings('ignore')


def run_backtest():
    """Детальний бектест з розширеною аналітикою"""
    initial_cash = 100000
    csv_path = os.path.join(os.path.dirname(__file__), "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv")

    # Завантаження даних
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Налаштування cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df.dropna()))
    cerebro.addstrategy(ProfitableBTCStrategy)
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(0.001)

    # Розширені аналізатори
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    print(f"💰 Стартовий капітал: ${initial_cash:,}")
    print("🚀 Агресивна стратегія для 1000%+ ROI...")

    results = cerebro.run()
    final_value = cerebro.broker.get_value()

    # Основні результати
    profit = final_value - initial_cash
    roi_percent = (profit / initial_cash) * 100
    annual_return = ((final_value / initial_cash) ** (1/7)) - 1

    # Детальний аналіз
    strategy = results[0]
    trades = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()

    # Статистика угод - захист від None значень
    total_trades = getattr(trades.get('total', {}), 'total', 0) or 0
    won_trades = getattr(trades.get('won', {}), 'total', 0) or 0
    lost_trades = getattr(trades.get('lost', {}), 'total', 0) or 0
    win_rate = (won_trades / max(total_trades, 1)) * 100

    # Детальна торгова статистика - захист від None
    won_total = getattr(trades.get('won', {}), 'pnl', {}).get('total', 0) or 0
    lost_total = abs(getattr(trades.get('lost', {}), 'pnl', {}).get('total', 0) or 0)
    avg_win = getattr(trades.get('won', {}), 'pnl', {}).get('average', 0) or 0
    avg_loss = getattr(trades.get('lost', {}), 'pnl', {}).get('average', 0) or 0

    profit_factor = (won_total / lost_total) if lost_total > 0 else float('inf')
    avg_trade = profit / max(total_trades, 1)

    # Аналіз просадок - захист від None
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0) or 0
    max_dd_period = drawdown.get('max', {}).get('len', 0) or 0

    # Sharpe ratio - захист від None
    sharpe_ratio = sharpe.get('sharperatio', 0) or 0

    # Виведення результатів
    print(f"\n📈 ОСНОВНІ РЕЗУЛЬТАТИ:")
    print(f"🎯 Кінцевий капітал: ${final_value:,.0f}")
    print(f"💵 Загальний прибуток: ${profit:+,.0f}")
    print(f"📊 ROI за 7 років: {roi_percent:+.1f}%")
    print(f"📅 Річна прибутковість: {annual_return*100:.1f}%")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")

    print(f"\n🎲 АНАЛІЗ УГОД:")
    print(f"Всього угод: {total_trades}")
    print(f"Прибуткових: {won_trades} ({win_rate:.1f}%)")
    print(f"Збиткових: {lost_trades} ({100-win_rate:.1f}%)")
    print(f"Середня угода: ${avg_trade:+,.0f}")
    print(f"Середній прибуток: ${avg_win:+,.0f}")
    print(f"Середній збиток: ${avg_loss:+,.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    print(f"\n📉 РИЗИК-АНАЛІЗ:")
    print(f"Макс. просадка: {max_drawdown:.1f}%")
    print(f"Тривалість просадки: {max_dd_period} днів")
    # Виправлено Risk/Reward розрахунок
    if avg_win > 0 and avg_loss != 0:
        risk_reward = abs(avg_loss/avg_win)
        print(f"Risk/Reward: {risk_reward:.2f}")
    else:
        print(f"Risk/Reward: N/A")

    # Основні результати угод
    print(f"\n📊 СТАТИСТИКА УГОД:")
    print(f"Всього угод: {total_trades}")
    print(f"Прибуткових: {won_trades} ({win_rate:.1f}%)")
    print(f"Збиткових: {lost_trades} ({100-win_rate:.1f}%)")
    print(f"Середня угода: ${avg_trade:+,.0f}")
    print(f"Середній прибуток: ${avg_win:+,.0f}")
    print(f"Середній збиток: ${avg_loss:+,.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    # Аналіз просадок
    print(f"\n📉 АНАЛІЗ ПРОСАДОК:")
    print(f"Макс. просадка: {max_drawdown:.1f}%")
    print(f"Тривалість просадки: {max_dd_period} днів")

    # Детальний аналіз угод
    print(f"\n🎲 ДЕТАЛЬНИЙ АНАЛІЗ УГОД:")
    print(f"Прибуткових угод: {won_trades}")
    print(f"Збиткових угод: {lost_trades}")
    print(f"Середній прибуток: ${avg_win:+,.0f}")
    print(f"Середній збиток: ${avg_loss:+,.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    # Оцінка проблем стратегії
    print(f"\n🔍 ДІАГНОСТИКА ПРОБЛЕМ:")

    if total_trades < 50:
        print(f"⚠️ Мало угод ({total_trades}) - стратегія занадто вибіркова")

    if win_rate < 60:
        print(f"⚠️ Низька точність ({win_rate:.1f}%) - потрібно покращити сигнали")

    if profit_factor < 1.5:
        print(f"⚠️ Низький Profit Factor ({profit_factor:.2f}) - збитки поглинають прибутки")

    if avg_trade < 0:
        print(f"⚠️ Негативна середня угода (${avg_trade:.0f})")

    if max_drawdown > 20:
        print(f"⚠️ Висока просадка ({max_drawdown:.1f}%) - занадто ризикова")

    if annual_return < 0.15:
        print(f"⚠️ Низька річна прибутковість ({annual_return*100:.1f}%)")

    # Порівняння з HODL
    btc_start = df.iloc[0]['close']
    btc_end = df.iloc[-1]['close']
    btc_return = ((btc_end / btc_start) - 1) * 100

    print(f"\n📋 ПОРІВНЯННЯ:")
    print(f"Bitcoin HODL: {btc_return:+.1f}%")
    print(f"Стратегія: {roi_percent:+.1f}%")
    print(f"Відставання: {roi_percent - btc_return:.1f}%")

    # Рекомендації
    print(f"\n💡 РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ:")
    print("1. Збільшити частоту угод (зменшити поріг сигналів)")
    print("2. Покращити співвідношення прибуток/збиток")
    print("3. Додати фільтри для зменшення хибних сигналів")
    print("4. Оптимізувати розмір позицій")
    print("5. Розглянути динамічні стоп-лоси")

    if roi_percent >= 1000:
        print(f"\n🎉 ЦІЛЬ ДОСЯГНУТА! ROI {roi_percent:.1f}% > 1000%")
    elif roi_percent >= 100:
        print(f"\n🔥 Хороший результат! ROI {roi_percent:.1f}%")
    else:
        print(f"\n⚠️ Потрібні кардинальні зміни для досягнення 1000%")


if __name__ == '__main__':
    try:
        run_backtest()
    except Exception as e:
        print(f"❌ Помилка: {e}")
