"""
Ультра-агрессивная конфигурация для достижения 300% годовой доходности.
Специально разработано для автоматической торговли 24/7 на 15-минутных интервалах.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .trading_config_15m import TradingConfig15m, DataManager15m


@dataclass
class UltraAggressiveConfig(TradingConfig15m):
    """Ультра-агрессивная конфигурация для экстремальной прибыльности."""
    
    # Базовые параметры для агрессивной торговли
    initial_balance: float = 100.0  # Начинаем с минимума по требованию
    commission_rate: float = 0.0003  # Оптимизированная комиссия для частой торговли
    slippage_rate: float = 0.0001  # Минимальное проскальзывание
    spread_rate: float = 0.00005  # Узкий спред для скальпинга
    
    # Экстремальное управление капиталом
    min_trade_amount: float = 5.0  # Позволяем мелкие сделки
    max_position_size: float = 1.0  # 100% капитала как требуется
    
    # Ультра-настройки ликвидности для скальпинга
    enable_partial_fills: bool = False  # Отключаем для скорости
    liquidity_impact_threshold: float = 0.0001
    max_order_size_ratio: float = 0.1
    
    # Агрессивные технические индикаторы
    include_technical_indicators: bool = True
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [3, 7, 15],  # Очень короткие периоды для скальпинга
        'ema': [3, 7, 15],
        'rsi': [5, 9],  # Сверх-отзывчивый RSI
        'macd': [5, 13, 8],  # Быстрый MACD
        'bollinger': [8],  # Короткие полосы Боллинджера
        'atr': [5, 10],  # Быстрый ATR
        'momentum': [2, 5, 8],  # Экстремально короткий momentum
        'stochastic': [5, 9],
        'williams_r': [5, 9],
        'vwap': []  # VWAP для текущего дня
    })
    
    # Окно наблюдения для скальпинга
    lookback_window: int = 16  # 4 часа истории (16 * 15мин)
    
    # Ультра-агрессивная схема наград
    reward_scheme: str = 'ultra_aggressive'
    
    # Настройки для 24/7 торговли
    enable_24_7_trading: bool = True
    risk_management_enabled: bool = True
    auto_stop_loss: float = 0.20  # Автоматический стоп-лосс при 20% просадке
    
    # Настройки производительности
    fast_execution: bool = True
    optimize_for_speed: bool = True


@dataclass 
class BTCUltraConfig(UltraAggressiveConfig):
    """Специализированная конфигурация для BTC торговли."""
    
    symbol: str = 'BTCUSDT'
    exchange: str = 'binance'
    
    # BTC-специфичные параметры
    commission_rate: float = 0.0001  # Меньше комиссия для BTC
    min_trade_amount: float = 8.0  # Чуть больше для BTC
    
    # Более консервативные индикаторы для BTC
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [5, 10, 20],
        'ema': [5, 10, 20], 
        'rsi': [7, 14],
        'macd': [8, 17, 9],
        'bollinger': [12],
        'atr': [7, 14],
        'momentum': [3, 7, 12],
        'stochastic': [7, 14],
        'williams_r': [7, 14],
        'vwap': []
    })
    
    reward_scheme: str = 'btc_specialized'


@dataclass
class ETHUltraConfig(UltraAggressiveConfig):
    """Специализированная конфигурация для ETH торговли."""
    
    symbol: str = 'ETHUSDT'
    exchange: str = 'binance'
    
    # ETH-специфичные параметры
    commission_rate: float = 0.0001
    min_trade_amount: float = 6.0
    
    # ETH более волатилен, чем BTC
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [3, 8, 16],
        'ema': [3, 8, 16],
        'rsi': [6, 12],  # Более отзывчивый для ETH
        'macd': [6, 15, 8],
        'bollinger': [10],
        'atr': [6, 12],
        'momentum': [2, 6, 10],
        'stochastic': [6, 12],
        'williams_r': [6, 12],
        'vwap': []
    })
    
    reward_scheme: str = 'btc_specialized'  # Используем ту же специализированную схему


class UltraAggressiveDataManager(DataManager15m):
    """Расширенный менеджер данных для ультра-агрессивной торговли."""
    
    @staticmethod
    def validate_ultra_aggressive_config(config: UltraAggressiveConfig) -> bool:
        """Валидация ультра-агрессивной конфигурации."""
        
        # Базовая валидация 15мин
        if not DataManager15m.validate_15m_config(config):
            return False
            
        # Проверка достаточности данных для агрессивной торговли
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            data_path = os.path.join(
                project_root, 'CryptoTrade', 'data', config.exchange,
                config.symbol, '15m', '2018_01_01-now.csv'
            )
            
            if os.path.exists(data_path):
                import pandas as pd
                df = pd.read_csv(data_path)
                
                # Для ультра-агрессивной торговли нужно много данных
                if len(df) < 50000:  # Минимум 50k записей для агрессивных стратегий
                    print(f"⚠️ Недостаточно данных для ультра-агрессивной торговли: {len(df)} записей")
                    print(f"💡 Рекомендуется минимум 50,000 записей")
                    return False
                    
                # Проверяем качество данных (нет больших пропусков)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Проверяем временные интервалы
                time_diffs = df['timestamp'].diff().dt.total_seconds() / 60  # в минутах
                expected_interval = 15
                large_gaps = time_diffs[time_diffs > expected_interval * 2]
                
                if len(large_gaps) > len(df) * 0.01:  # Более 1% пропусков
                    print(f"⚠️ Обнаружены пропуски в данных: {len(large_gaps)} больших интервалов")
                    print(f"💡 Качество данных может влиять на агрессивные стратегии")
                
                print(f"✅ Ультра-агрессивная конфигурация: {len(df)} записей, качество данных проверено")
                
        except Exception as e:
            print(f"❌ Ошибка валидации данных: {e}")
            return False
            
        # Проверка параметров для экстремальной торговли
        if config.auto_stop_loss > 0.25:
            print(f"⚠️ Стоп-лосс {config.auto_stop_loss:.1%} слишком высок для ультра-агрессивной торговли")
            print(f"💡 Рекомендуется максимум 20%")
            
        if config.initial_balance < 50:
            print(f"⚠️ Начальный баланс {config.initial_balance} очень низкий")
            print(f"💡 Минимум $100 рекомендуется для стабильной торговли")
        
        return True
    
    @staticmethod
    def get_optimal_pairs_for_aggressive_trading() -> List[Dict]:
        """Получить оптимальные пары для агрессивной торговли."""
        stats = DataManager15m.get_15m_data_stats()
        
        # Критерии для агрессивной торговли
        optimal_pairs = []
        
        for pair_key, data in stats.items():
            if data['quality'] == 'good' and data['records'] >= 50000:
                exchange, symbol = pair_key.split('_', 1)
                
                # Приоритет для крупных ликвидных пар
                priority_score = 0
                
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    priority_score += 10  # Максимальный приоритет
                elif symbol in ['BNBUSDT', 'SOLUSDT', 'XRPUSDT']:
                    priority_score += 7  # Высокий приоритет
                elif 'USDT' in symbol:
                    priority_score += 5  # Средний приоритет для USDT пар
                elif 'USDC' in symbol:
                    priority_score += 3  # Низкий приоритет для USDC пар
                
                # Бонус для биржи Binance
                if exchange == 'binance':
                    priority_score += 3
                
                optimal_pairs.append({
                    'exchange': exchange,
                    'symbol': symbol,
                    'records': data['records'],
                    'priority_score': priority_score,
                    'start_date': data['start_date'],
                    'end_date': data['end_date']
                })
        
        # Сортируем по приоритету
        optimal_pairs.sort(key=lambda x: x['priority_score'], reverse=True)
        return optimal_pairs
    
    @staticmethod
    def create_multi_pair_configs(target_pairs: List[str] = None) -> List[UltraAggressiveConfig]:
        """Создать конфигурации для нескольких пар одновременно."""
        if target_pairs is None:
            target_pairs = ['BTCUSDT', 'ETHUSDT']
            
        configs = []
        optimal_pairs = UltraAggressiveDataManager.get_optimal_pairs_for_aggressive_trading()
        
        for pair_info in optimal_pairs:
            if pair_info['symbol'] in target_pairs:
                if pair_info['symbol'] == 'BTCUSDT':
                    config = BTCUltraConfig(
                        exchange=pair_info['exchange'],
                        initial_balance=100.0
                    )
                elif pair_info['symbol'] == 'ETHUSDT':
                    config = ETHUltraConfig(
                        exchange=pair_info['exchange'], 
                        initial_balance=100.0
                    )
                else:
                    config = UltraAggressiveConfig(
                        exchange=pair_info['exchange'],
                        symbol=pair_info['symbol'],
                        initial_balance=100.0
                    )
                
                if UltraAggressiveDataManager.validate_ultra_aggressive_config(config):
                    configs.append(config)
        
        return configs


def create_ultra_aggressive_btc_config() -> BTCUltraConfig:
    """Создать ультра-агрессивную BTC конфигурацию."""
    return BTCUltraConfig(initial_balance=100.0)


def create_ultra_aggressive_eth_config() -> ETHUltraConfig:
    """Создать ультра-агрессивную ETH конфигурацию.""" 
    return ETHUltraConfig(initial_balance=100.0)


def interactive_ultra_aggressive_creator() -> UltraAggressiveConfig:
    """Интерактивное создание ультра-агрессивной конфигурации."""
    print("🔥" + "="*70 + "🔥")
    print("   УЛЬТРА-АГРЕССИВНАЯ КОНФИГУРАЦИЯ ДЛЯ 300% ГОДОВЫХ")
    print("🔥" + "="*70 + "🔥")
    print()
    print("🎯 Цели:")
    print("   • Годовая доходность: 300%+")
    print("   • Максимальная просадка: 20%")
    print("   • Win rate: >60% (цель >70%)")
    print("   • Автоматическая торговля 24/7")
    print("   • Скальпинг на 15-минутных интервалах")
    print()
    
    # Показываем оптимальные пары
    print("📊 Анализ оптимальных пар для агрессивной торговли...")
    optimal_pairs = UltraAggressiveDataManager.get_optimal_pairs_for_aggressive_trading()
    
    print(f"\n🎯 Топ пары для ультра-агрессивной торговли:")
    for i, pair in enumerate(optimal_pairs[:8], 1):
        priority_emoji = "🔥" if pair['priority_score'] >= 10 else "⚡" if pair['priority_score'] >= 7 else "💎"
        print(f"   {i}. {priority_emoji} {pair['exchange']}:{pair['symbol']} "
              f"({pair['records']:,} записей, приоритет: {pair['priority_score']})")
    
    # Выбор конфигурации
    print(f"\n🔥 Режимы ультра-агрессивной торговли:")
    print(f"   1. 🟡 BTC Ультра-Агрессив (специализированная схема)")
    print(f"   2. 🔷 ETH Ультра-Агрессив (специализированная схема)")
    print(f"   3. ⚡ Произвольная пара (ультра-агрессивная схема)")
    print(f"   4. 🎯 Интерактивный выбор")
    
    while True:
        try:
            choice = int(input(f"\nВыберите режим (1-4): "))
            if choice == 1:
                config = create_ultra_aggressive_btc_config()
                break
            elif choice == 2:
                config = create_ultra_aggressive_eth_config() 
                break
            elif choice == 3:
                # Выбор произвольной пары
                print(f"\nВыберите пару из топа:")
                for i, pair in enumerate(optimal_pairs[:5], 1):
                    print(f"   {i}. {pair['exchange']}:{pair['symbol']}")
                
                while True:
                    try:
                        pair_choice = int(input(f"Выберите пару (1-5): ")) - 1
                        if 0 <= pair_choice < min(5, len(optimal_pairs)):
                            selected = optimal_pairs[pair_choice]
                            config = UltraAggressiveConfig(
                                exchange=selected['exchange'],
                                symbol=selected['symbol'],
                                initial_balance=100.0
                            )
                            break
                    except ValueError:
                        pass
                    print("❌ Неверный выбор!")
                break
            elif choice == 4:
                # Полностью интерактивный выбор
                from .trading_config_15m import interactive_15m_config_creator
                base_config = interactive_15m_config_creator()
                if base_config:
                    config = UltraAggressiveConfig(**base_config.__dict__)
                    config.reward_scheme = 'ultra_aggressive'
                else:
                    return None
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Настройка начального капитала
    print(f"\n💰 Начальный капитал:")
    print(f"   1. Минимальный ($100) - как требуется")
    print(f"   2. Средний ($500)")
    print(f"   3. Высокий ($1000)")
    print(f"   4. Пользовательский")
    
    while True:
        try:
            capital_choice = int(input("Выберите капитал (1-4): "))
            if capital_choice == 1:
                config.initial_balance = 100.0
                break
            elif capital_choice == 2:
                config.initial_balance = 500.0
                break
            elif capital_choice == 3:
                config.initial_balance = 1000.0
                break
            elif capital_choice == 4:
                config.initial_balance = float(input("Введите сумму USDT: "))
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Финальная валидация
    print(f"\n🔥 Создана ультра-агрессивная конфигурация:")
    print(f"   Биржа: {config.exchange}")
    print(f"   Пара: {config.symbol}")
    print(f"   Капитал: ${config.initial_balance}")
    print(f"   Таймфрейм: {config.timeframe}")
    print(f"   Схема наград: {config.reward_scheme}")
    print(f"   Цель: 300% годовых при просадке <20%")
    
    if UltraAggressiveDataManager.validate_ultra_aggressive_config(config):
        print("✅ Конфигурация готова для экстремальной торговли!")
        return config
    else:
        print("❌ Ошибка в ультра-агрессивной конфигурации!")
        return None


if __name__ == "__main__":
    print("🔥 Анализ данных для ультра-агрессивной торговли")
    optimal_pairs = UltraAggressiveDataManager.get_optimal_pairs_for_aggressive_trading()
    
    print(f"\n📊 Найдено {len(optimal_pairs)} оптимальных пар:")
    for pair in optimal_pairs[:10]:
        priority_emoji = "🔥" if pair['priority_score'] >= 10 else "⚡" if pair['priority_score'] >= 7 else "💎"
        print(f"{priority_emoji} {pair['exchange']}:{pair['symbol']} - {pair['records']:,} записей")