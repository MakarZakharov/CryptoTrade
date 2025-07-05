"""
Специализированная конфигурация для торговли на 15-минутных таймфреймах.
Оптимизирована для высокочастотной прибыльной торговли.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .trading_config import TradingConfig, DataManager


@dataclass
class TradingConfig15m(TradingConfig):
    """Конфигурация для 15-минутной торговли."""
    
    # Данные
    timeframe: str = '15m'
    
    # Оптимизированные торговые параметры для 15мин
    initial_balance: float = 1000.0  # Больше капитала для стабильности
    commission_rate: float = 0.0005  # Более низкая комиссия (0.05%)
    slippage_rate: float = 0.0002  # Меньшее проскальзывание для 15мин
    spread_rate: float = 0.0001  # Более узкий спред
    
    # Управление капиталом для частой торговли
    min_trade_amount: float = 50.0  # Больше минимум для качественных сделок
    max_position_size: float = 0.8  # 80% максимум для управления рисками
    
    # Улучшенные настройки ликвидности
    enable_partial_fills: bool = True
    liquidity_impact_threshold: float = 0.0005  # Меньше влияние на ликвидность
    max_order_size_ratio: float = 0.05  # 5% от объема максимум
    
    # Оптимизированные технические индикаторы для 15мин
    include_technical_indicators: bool = True
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [5, 10, 20],  # Более короткие периоды
        'ema': [5, 10, 20],
        'rsi': [7, 14],  # Более отзывчивый RSI
        'macd': [8, 17, 9],  # Более быстрый MACD
        'bollinger': [10],  # Более короткий период
        'atr': [7, 14],  # Более отзывчивый ATR
        'adx': [7, 14],
        'momentum': [3, 5, 10],  # Более короткие периоды
        'stochastic': [7, 14],
        'williams_r': [7, 14],
        'obv': [],
        'vwap': []
    })
    
    # Обучение для 15мин (меньше окно для быстрых решений)
    lookback_window: int = 24  # 6 часов истории
    
    # Награды оптимизированы для частой торговли
    reward_scheme: str = 'optimized'


def create_15m_config(symbol: str, exchange: str = 'binance') -> TradingConfig15m:
    """Создать оптимизированную конфигурацию для 15мин торговли."""
    return TradingConfig15m(
        exchange=exchange,
        symbol=symbol,
        timeframe='15m',
        reward_scheme='optimized',
        initial_balance=1000.0
    )


def get_popular_15m_pairs() -> List[str]:
    """Получить популярные пары для 15мин торговли."""
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
        'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT'
    ]


class DataManager15m(DataManager):
    """Специализированный менеджер данных для 15мин."""
    
    @staticmethod
    def validate_15m_config(config: TradingConfig15m) -> bool:
        """Проверить конфигурацию для 15мин торговли."""
        # Проверяем базовую валидность
        if not DataManager.validate_config(config):
            return False
        
        # Проверяем наличие 15мин данных
        available_timeframes = DataManager.get_available_timeframes(
            config.exchange, config.symbol
        )
        
        if '15m' not in available_timeframes:
            print(f"❌ 15-минутные данные недоступны для {config.symbol}")
            print(f"💡 Доступные таймфреймы: {available_timeframes}")
            return False
        
        # Проверяем размер данных (должно быть достаточно для обучения)
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            data_path = os.path.join(
                project_root, 'CryptoTrade', 'data', config.exchange, 
                config.symbol, '15m', '2018_01_01-now.csv'
            )
            
            if os.path.exists(data_path):
                import pandas as pd
                df = pd.read_csv(data_path)
                if len(df) < 10000:  # Минимум 10k записей
                    print(f"⚠️ Недостаточно 15мин данных: {len(df)} записей")
                    print(f"💡 Рекомендуется минимум 10,000 записей")
                    return False
                print(f"✅ 15мин данные: {len(df)} записей")
        except Exception as e:
            print(f"❌ Ошибка проверки данных: {e}")
            return False
        
        return True
    
    @staticmethod
    def get_15m_data_stats() -> Dict[str, Dict]:
        """Получить статистику по 15мин данным."""
        stats = {}
        available_pairs = DataManager.get_available_pairs()
        
        for exchange, pairs in available_pairs.items():
            for pair in pairs:
                timeframes = DataManager.get_available_timeframes(exchange, pair)
                if '15m' in timeframes:
                    try:
                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                        data_path = os.path.join(
                            project_root, 'CryptoTrade', 'data', exchange, 
                            pair, '15m', '2018_01_01-now.csv'
                        )
                        
                        if os.path.exists(data_path):
                            import pandas as pd
                            df = pd.read_csv(data_path)
                            
                            stats[f"{exchange}_{pair}"] = {
                                'records': len(df),
                                'start_date': df['timestamp'].iloc[0] if len(df) > 0 else None,
                                'end_date': df['timestamp'].iloc[-1] if len(df) > 0 else None,
                                'quality': 'good' if len(df) >= 10000 else 'insufficient'
                            }
                    except Exception:
                        continue
        
        return stats


def interactive_15m_config_creator() -> TradingConfig15m:
    """Интерактивное создание конфигурации для 15мин торговли."""
    print("=== Создание конфигурации для 15-минутной торговли ===\n")
    
    # Показываем статистику доступных данных
    print("📊 Проверка доступных 15мин данных...")
    stats = DataManager15m.get_15m_data_stats()
    
    good_pairs = []
    insufficient_pairs = []
    
    for pair_key, data in stats.items():
        exchange, symbol = pair_key.split('_', 1)
        if data['quality'] == 'good':
            good_pairs.append((exchange, symbol, data['records']))
        else:
            insufficient_pairs.append((exchange, symbol, data['records']))
    
    if not good_pairs:
        print("❌ Недостаточно качественных 15мин данных для обучения!")
        print("💡 Рекомендуется сначала загрузить больше данных")
        return None
    
    print(f"\n✅ Найдено {len(good_pairs)} пар с качественными 15мин данными:")
    for i, (exchange, symbol, records) in enumerate(good_pairs[:10], 1):
        print(f"  {i}. {exchange}:{symbol} ({records:,} записей)")
    
    # Выбор пары
    while True:
        try:
            choice = int(input(f"\nВыберите пару (1-{min(10, len(good_pairs))}): ")) - 1
            if 0 <= choice < min(10, len(good_pairs)):
                selected_exchange, selected_symbol, _ = good_pairs[choice]
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Выбор начального капитала
    print("\nНачальный капитал для 15мин торговли:")
    print("   1. Консервативный (500 USDT)")
    print("   2. Стандартный (1,000 USDT)")  
    print("   3. Агрессивный (2,000 USDT)")
    print("   4. Пользовательский")
    
    while True:
        try:
            choice = int(input("Выберите (1-4): "))
            if choice == 1:
                initial_balance = 500.0
                break
            elif choice == 2:
                initial_balance = 1000.0
                break
            elif choice == 3:
                initial_balance = 2000.0
                break
            elif choice == 4:
                initial_balance = float(input("Введите сумму USDT: "))
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Создаем конфигурацию
    config = TradingConfig15m(
        exchange=selected_exchange,
        symbol=selected_symbol,
        timeframe='15m',
        reward_scheme='optimized',
        initial_balance=initial_balance
    )
    
    print(f"\n=== Создана конфигурация для 15мин торговли ===")
    print(f"Биржа: {config.exchange}")
    print(f"Пара: {config.symbol}")
    print(f"Таймфрейм: {config.timeframe}")
    print(f"Начальный капитал: {config.initial_balance} USDT")
    print(f"Схема наград: {config.reward_scheme} (оптимизированная)")
    print(f"Окно наблюдения: {config.lookback_window} (6 часов)")
    print(f"Комиссия: {config.commission_rate*100:.3f}%")
    
    # Проверяем валидность
    if DataManager15m.validate_15m_config(config):
        print("✅ Конфигурация готова для 15мин торговли!")
        return config
    else:
        print("❌ Ошибка в конфигурации!")
        return None


if __name__ == "__main__":
    # Показать статистику 15мин данных
    print("📊 Статистика 15-минутных данных:")
    stats = DataManager15m.get_15m_data_stats()
    
    for pair_key, data in list(stats.items())[:10]:
        exchange, symbol = pair_key.split('_', 1)
        quality_emoji = "✅" if data['quality'] == 'good' else "⚠️"
        print(f"{quality_emoji} {exchange}:{symbol} - {data['records']:,} записей")