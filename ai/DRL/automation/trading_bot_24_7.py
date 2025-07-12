"""
24/7 Автоматическая торговая система для ультра-агрессивного DRL агента.
Полностью автономная торговля с мониторингом рисков в реальном времени.
"""

import os
import sys
import time
import threading
import queue
import logging
import signal
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from CryptoTrade.ai.DRL.config.ultra_aggressive_config import (
    UltraAggressiveConfig, BTCUltraConfig, ETHUltraConfig,
    UltraAggressiveDataManager
)
from CryptoTrade.ai.DRL.environment.trading_env import TradingEnv
from CryptoTrade.ai.DRL.agents.ultra_aggressive_ppo_agent import UltraAggressivePPOAgent


class TradingBotStatus(Enum):
    """Статусы торгового бота."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskMetrics:
    """Метрики рисков для мониторинга."""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    current_return: float = 0.0
    win_rate: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    portfolio_value: float = 0.0
    last_update: datetime = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.last_update:
            result['last_update'] = self.last_update.isoformat()
        return result


@dataclass
class TradingSession:
    """Торговая сессия."""
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    session_id: str = ""
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


class RiskManager:
    """Менеджер рисков для 24/7 торговли."""
    
    def __init__(self, 
                 max_drawdown: float = 0.20,
                 max_consecutive_losses: int = 10,
                 max_trades_per_day: int = 500,
                 min_win_rate: float = 0.40,
                 emergency_stop_drawdown: float = 0.25):
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_trades_per_day = max_trades_per_day
        self.min_win_rate = min_win_rate
        self.emergency_stop_drawdown = emergency_stop_drawdown
        
        self.current_metrics = RiskMetrics()
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.risk_alerts = []
        
    def update_metrics(self, portfolio_value: float, trade_history: List[Dict], 
                      initial_balance: float) -> None:
        """Обновить метрики рисков."""
        current_return = (portfolio_value - initial_balance) / initial_balance
        
        # Расчет просадки
        peak_value = max(initial_balance, portfolio_value)
        current_drawdown = (peak_value - portfolio_value) / peak_value
        
        # Обновление дневных торгов
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = today
        
        # Подсчет торгов за сегодня
        today_trades = [t for t in trade_history 
                       if t.get('timestamp', datetime.now()).date() == today]
        self.daily_trades = len(today_trades)
        
        # Win rate
        if trade_history:
            profitable = sum(1 for t in trade_history if t.get('profit', 0) > 0)
            win_rate = profitable / len(trade_history)
        else:
            win_rate = 0.0
        
        # Consecutive losses
        consecutive_losses = 0
        for trade in reversed(trade_history):
            if trade.get('profit', 0) <= 0:
                consecutive_losses += 1
            else:
                break
        
        # Обновляем метрики
        self.current_metrics.current_drawdown = current_drawdown
        self.current_metrics.max_drawdown = max(self.current_metrics.max_drawdown, current_drawdown)
        self.current_metrics.current_return = current_return
        self.current_metrics.win_rate = win_rate
        self.current_metrics.trades_today = self.daily_trades
        self.current_metrics.consecutive_losses = consecutive_losses
        self.current_metrics.portfolio_value = portfolio_value
        self.current_metrics.last_update = datetime.now()
    
    def check_risks(self) -> List[str]:
        """Проверить риски и вернуть предупреждения."""
        alerts = []
        
        # Критические риски
        if self.current_metrics.current_drawdown > self.emergency_stop_drawdown:
            alerts.append(f"🚨 КРИТИЧЕСКАЯ ПРОСАДКА: {self.current_metrics.current_drawdown:.2%}")
        elif self.current_metrics.current_drawdown > self.max_drawdown:
            alerts.append(f"⚠️ ПРЕВЫШЕНА ПРОСАДКА: {self.current_metrics.current_drawdown:.2%}")
        
        if self.current_metrics.consecutive_losses > self.max_consecutive_losses:
            alerts.append(f"⚠️ МНОГО УБЫТОЧНЫХ СДЕЛОК ПОДРЯД: {self.current_metrics.consecutive_losses}")
        
        if self.daily_trades > self.max_trades_per_day:
            alerts.append(f"⚠️ СЛИШКОМ МНОГО СДЕЛОК ЗА ДЕНЬ: {self.daily_trades}")
        
        if len(self.current_metrics.__dict__) > 10 and self.current_metrics.win_rate < self.min_win_rate:
            alerts.append(f"⚠️ НИЗКИЙ WIN RATE: {self.current_metrics.win_rate:.1%}")
        
        self.risk_alerts = alerts
        return alerts
    
    def should_emergency_stop(self) -> bool:
        """Нужна ли экстренная остановка."""
        return self.current_metrics.current_drawdown > self.emergency_stop_drawdown
    
    def should_pause_trading(self) -> bool:
        """Нужна ли пауза в торговле."""
        alerts = self.check_risks()
        critical_alerts = [a for a in alerts if a.startswith("🚨")]
        return len(critical_alerts) > 0


class TradingBot24x7:
    """Главный класс 24/7 торгового бота."""
    
    def __init__(self, 
                 config: UltraAggressiveConfig,
                 model_path: str,
                 log_level: int = logging.INFO,
                 save_state_frequency: int = 300):  # 5 минут
        
        self.config = config
        self.model_path = model_path
        self.save_state_frequency = save_state_frequency
        
        # Настройка логирования
        self._setup_logging(log_level)
        
        # Состояние бота
        self.status = TradingBotStatus.STOPPED
        self.running = False
        self.paused = False
        
        # Торговые компоненты
        self.agent = None
        self.env = None
        self.risk_manager = RiskManager(
            max_drawdown=config.auto_stop_loss,
            max_consecutive_losses=15,
            max_trades_per_day=200,  # Разумное ограничение для 15мин
            min_win_rate=0.45
        )
        
        # Мониторинг и статистика
        self.current_session = None
        self.session_history = []
        self.performance_queue = queue.Queue()
        
        # Потоки
        self.trading_thread = None
        self.monitoring_thread = None
        self.state_saving_thread = None
        
        # Пути для сохранения
        self.bot_data_dir = f"CryptoTrade/ai/DRL/automation/bot_data"
        os.makedirs(self.bot_data_dir, exist_ok=True)
        
        # Обработка сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🤖 Торговый бот 24/7 инициализирован")
    
    def _setup_logging(self, log_level: int):
        """Настройка системы логирования."""
        log_dir = "CryptoTrade/ai/DRL/automation/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("TradingBot24x7")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown."""
        self.logger.info(f"Получен сигнал {signum}, начинаю остановку бота...")
        self.stop()
    
    def initialize(self) -> bool:
        """Инициализация торговых компонентов."""
        try:
            self.status = TradingBotStatus.STARTING
            self.logger.info("🔄 Инициализация торговых компонентов...")
            
            # Создание среды
            self.env = TradingEnv(self.config)
            self.logger.info(f"✅ Среда создана: {self.config.symbol} на {self.config.timeframe}")
            
            # Создание и загрузка агента
            self.agent = UltraAggressivePPOAgent(self.config, use_gpu=True, multi_env=False)
            
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Модель не найдена: {self.model_path}")
                return False
            
            self.agent.load(self.model_path, self.env)
            self.logger.info(f"✅ Агент загружен: {self.model_path}")
            
            # Создание новой торговой сессии
            self._start_new_session()
            
            self.status = TradingBotStatus.RUNNING
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации: {e}")
            self.status = TradingBotStatus.ERROR
            return False
    
    def _start_new_session(self):
        """Начать новую торговую сессию."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TradingSession(
            start_time=datetime.now(),
            initial_balance=self.config.initial_balance,
            session_id=session_id
        )
        self.logger.info(f"🆕 Новая торговая сессия: {session_id}")
    
    def start(self) -> bool:
        """Запустить торгового бота."""
        if self.status == TradingBotStatus.RUNNING:
            self.logger.warning("⚠️ Бот уже запущен")
            return True
        
        if not self.initialize():
            return False
        
        self.running = True
        self.paused = False
        
        # Запуск потоков
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.state_saving_thread = threading.Thread(target=self._state_saving_loop, daemon=True)
        
        self.trading_thread.start()
        self.monitoring_thread.start()
        self.state_saving_thread.start()
        
        self.logger.info("🚀 Торговый бот 24/7 запущен!")
        return True
    
    def stop(self):
        """Остановить торгового бота."""
        self.logger.info("⏹️ Остановка торгового бота...")
        
        self.running = False
        self.status = TradingBotStatus.STOPPED
        
        # Завершение текущей сессии
        if self.current_session:
            self._end_current_session()
        
        # Ожидание завершения потоков
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        if self.state_saving_thread and self.state_saving_thread.is_alive():
            self.state_saving_thread.join(timeout=5)
        
        self.logger.info("✅ Торговый бот остановлен")
    
    def pause(self):
        """Приостановить торговлю."""
        self.paused = True
        self.status = TradingBotStatus.PAUSED
        self.logger.info("⏸️ Торговля приостановлена")
    
    def resume(self):
        """Возобновить торговлю."""
        self.paused = False
        self.status = TradingBotStatus.RUNNING
        self.logger.info("▶️ Торговля возобновлена")
    
    def _trading_loop(self):
        """Основной цикл торговли."""
        self.logger.info("🔄 Запуск торгового цикла...")
        
        obs = self.env.reset()
        last_action_time = time.time()
        
        while self.running:
            try:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Проверка рисков
                if self.risk_manager.should_emergency_stop():
                    self.logger.critical("🚨 ЭКСТРЕННАЯ ОСТАНОВКА ПО РИСКАМ!")
                    self.status = TradingBotStatus.EMERGENCY_STOP
                    break
                
                if self.risk_manager.should_pause_trading():
                    self.logger.warning("⚠️ Торговля приостановлена по рискам")
                    self.pause()
                    time.sleep(60)  # Пауза на минуту
                    continue
                
                # Получение действия от агента
                action = self.agent.act(obs)
                
                # Выполнение действия в среде
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Обновление метрик рисков
                self.risk_manager.update_metrics(
                    portfolio_value=info['portfolio_value'],
                    trade_history=info.get('trade_history', []),
                    initial_balance=self.current_session.initial_balance
                )
                
                # Обновление сессии
                self._update_session(info)
                
                # Логирование торговых действий
                if abs(action[0]) > 0.05:  # Значимое действие
                    action_type = "BUY" if action[0] > 0 else "SELL"
                    self.logger.info(
                        f"📈 {action_type}: {abs(action[0])*100:.1f}% | "
                        f"Портфель: ${info['portfolio_value']:.2f} | "
                        f"Доходность: {info['total_return']:.2%} | "
                        f"Просадка: {self.risk_manager.current_metrics.current_drawdown:.2%}"
                    )
                
                # Если эпизод завершен, перезапускаем
                if done or truncated:
                    self.logger.info("🔄 Эпизод завершен, перезапуск...")
                    obs = self.env.reset()
                    self._end_current_session()
                    self._start_new_session()
                
                # Задержка между действиями (15 минут для реальной торговли)
                time.sleep(max(0, 900 - (time.time() - last_action_time)))  # 900 сек = 15 мин
                last_action_time = time.time()
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка в торговом цикле: {e}")
                time.sleep(10)  # Пауза при ошибке
        
        self.logger.info("⏹️ Торговый цикл завершен")
    
    def _monitoring_loop(self):
        """Цикл мониторинга производительности."""
        self.logger.info("📊 Запуск мониторинга...")
        
        while self.running:
            try:
                if not self.paused and self.risk_manager.current_metrics.last_update:
                    # Проверка рисков
                    alerts = self.risk_manager.check_risks()
                    
                    if alerts:
                        for alert in alerts:
                            self.logger.warning(alert)
                    
                    # Периодическая отчетность
                    if int(time.time()) % 3600 == 0:  # Каждый час
                        self._log_hourly_report()
                
                time.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка мониторинга: {e}")
                time.sleep(60)
        
        self.logger.info("📊 Мониторинг завершен")
    
    def _state_saving_loop(self):
        """Цикл сохранения состояния."""
        while self.running:
            try:
                self._save_bot_state()
                time.sleep(self.save_state_frequency)
            except Exception as e:
                self.logger.error(f"❌ Ошибка сохранения состояния: {e}")
                time.sleep(self.save_state_frequency)
    
    def _update_session(self, info: Dict):
        """Обновить текущую сессию."""
        if not self.current_session:
            return
        
        self.current_session.final_balance = info['portfolio_value']
        self.current_session.total_trades = info['total_trades']
        self.current_session.profitable_trades = int(info['total_trades'] * info['win_rate'])
        self.current_session.max_drawdown = self.risk_manager.current_metrics.max_drawdown
        self.current_session.total_return = info['total_return']
    
    def _end_current_session(self):
        """Завершить текущую сессию."""
        if not self.current_session:
            return
        
        self.current_session.end_time = datetime.now()
        self.session_history.append(self.current_session)
        
        duration = self.current_session.end_time - self.current_session.start_time
        self.logger.info(
            f"📋 Сессия завершена: {self.current_session.session_id} | "
            f"Длительность: {duration} | "
            f"Доходность: {self.current_session.total_return:.2%} | "
            f"Сделок: {self.current_session.total_trades}"
        )
    
    def _log_hourly_report(self):
        """Логировать часовой отчет."""
        metrics = self.risk_manager.current_metrics
        
        self.logger.info("=" * 60)
        self.logger.info("📊 ЧАСОВОЙ ОТЧЕТ")
        self.logger.info(f"   💰 Портфель: ${metrics.portfolio_value:.2f}")
        self.logger.info(f"   📈 Доходность: {metrics.current_return:.2%}")
        self.logger.info(f"   📉 Просадка: {metrics.current_drawdown:.2%}")
        self.logger.info(f"   🏆 Win Rate: {metrics.win_rate:.1%}")
        self.logger.info(f"   📊 Сделок сегодня: {metrics.trades_today}")
        self.logger.info(f"   ⚠️ Статус: {self.status.value}")
        self.logger.info("=" * 60)
    
    def _save_bot_state(self):
        """Сохранить состояние бота."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'status': self.status.value,
            'config': asdict(self.config),
            'risk_metrics': self.risk_manager.current_metrics.to_dict(),
            'current_session': self.current_session.to_dict() if self.current_session else None,
            'session_count': len(self.session_history)
        }
        
        state_file = os.path.join(self.bot_data_dir, 'bot_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_status(self) -> Dict:
        """Получить текущий статус бота."""
        return {
            'status': self.status.value,
            'running': self.running,
            'paused': self.paused,
            'risk_metrics': self.risk_manager.current_metrics.to_dict(),
            'current_session': self.current_session.to_dict() if self.current_session else None,
            'alerts': self.risk_manager.risk_alerts
        }


def main():
    """Главная функция для запуска 24/7 бота."""
    print("🤖" + "="*70 + "🤖")
    print("   АВТОМАТИЧЕСКИЙ ТОРГОВЫЙ БОТ 24/7")
    print("   Ультра-агрессивная торговля криптовалютами")
    print("🤖" + "="*70 + "🤖")
    
    # Настройка
    print("\n⚙️ Настройка бота...")
    
    # Выбор конфигурации
    print("Выберите конфигурацию:")
    print("  1. BTC Ultra Aggressive")
    print("  2. ETH Ultra Aggressive")
    print("  3. Пользовательская")
    
    choice = input("Выбор (1-3): ").strip()
    
    if choice == "1":
        config = BTCUltraConfig(initial_balance=100.0)
    elif choice == "2":
        config = ETHUltraConfig(initial_balance=100.0)
    else:
        # Пользовательская конфигурация
        symbol = input("Введите символ (например, BTCUSDT): ").upper()
        balance = float(input("Введите начальный баланс USDT: "))
        config = UltraAggressiveConfig(symbol=symbol, initial_balance=balance)
    
    # Выбор модели
    print(f"\n🔍 Поиск обученных моделей для {config.symbol}...")
    
    models_dir = "CryptoTrade/ai/DRL/models"
    available_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, item)
            if os.path.isdir(model_dir):
                # Ищем файлы моделей
                for model_file in ['best_model.zip', 'final_model.zip']:
                    model_path = os.path.join(model_dir, model_file)
                    if os.path.exists(model_path):
                        available_models.append({
                            'name': f"{item}/{model_file}",
                            'path': model_path,
                            'size': os.path.getsize(model_path),
                            'modified': os.path.getmtime(model_path)
                        })
    
    if not available_models:
        print("❌ Модели не найдены!")
        print("💡 Сначала обучите модель с помощью mvp_train.py")
        return
    
    print(f"\nДоступные модели:")
    for i, model in enumerate(available_models, 1):
        size_mb = model['size'] / (1024 * 1024)
        modified = datetime.fromtimestamp(model['modified']).strftime('%Y-%m-%d %H:%M')
        print(f"  {i}. {model['name']} ({size_mb:.1f} MB, {modified})")
    
    while True:
        try:
            model_choice = int(input(f"Выберите модель (1-{len(available_models)}): ")) - 1
            if 0 <= model_choice < len(available_models):
                model_path = available_models[model_choice]['path']
                break
        except ValueError:
            pass
        print("❌ Неверный выбор!")
    
    # Создание и запуск бота
    print(f"\n🤖 Создание торгового бота...")
    print(f"   Пара: {config.symbol}")
    print(f"   Баланс: ${config.initial_balance}")
    print(f"   Модель: {model_path}")
    
    bot = TradingBot24x7(config, model_path, log_level=logging.INFO)
    
    print(f"\n🚀 Запуск бота...")
    if bot.start():
        print(f"✅ Бот успешно запущен!")
        print(f"📊 Мониторинг: tail -f CryptoTrade/ai/DRL/automation/logs/trading_bot_*.log")
        print(f"⏹️ Для остановки: Ctrl+C")
        
        try:
            # Основной цикл
            while bot.running:
                time.sleep(1)
                
                # Простой интерактивный интерфейс
                try:
                    # Неблокирующий ввод (упрощенная версия)
                    import select
                    import sys
                    
                    if sys.platform != 'win32':
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            command = sys.stdin.readline().strip().lower()
                            if command == 'status':
                                status = bot.get_status()
                                print(f"\n📊 Статус: {status['status']}")
                                if status['risk_metrics']:
                                    metrics = status['risk_metrics']
                                    print(f"💰 Портфель: ${metrics['portfolio_value']:.2f}")
                                    print(f"📈 Доходность: {metrics['current_return']:.2%}")
                                    print(f"📉 Просадка: {metrics['current_drawdown']:.2%}")
                            elif command == 'pause':
                                bot.pause()
                            elif command == 'resume':
                                bot.resume()
                            elif command == 'stop':
                                break
                except:
                    pass  # Игнорируем ошибки ввода
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Получен сигнал остановки...")
        
        bot.stop()
    else:
        print(f"❌ Не удалось запустить бота!")


if __name__ == "__main__":
    main()