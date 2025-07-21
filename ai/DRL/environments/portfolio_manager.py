"""Менеджер портфеля для торговой среды."""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass

from ..config import TradingConfig
from ..utils import DRLLogger


@dataclass
class Trade:
    """Информация о сделке."""
    timestamp: str
    action: str  # "buy", "sell", "hold"
    amount: float  # Количество криптовалюты
    price: float  # Цена исполнения
    value: float  # Общая стоимость сделки
    commission: float  # Комиссия
    pnl: float = 0.0  # Прибыль/убыток (для закрытых позиций)
    position_size_before: float = 0.0  # Размер позиции до сделки
    position_size_after: float = 0.0  # Размер позиции после сделки


class PortfolioManager:
    """
    Менеджер портфеля для управления торговыми операциями.
    
    Отвечает за:
    - Управление балансом USDT и криптовалют
    - Выполнение торговых операций
    - Расчет P&L и метрик портфеля
    - Контроль рисков
    """
    
    def __init__(self, config: TradingConfig, logger: Optional[DRLLogger] = None):
        """
        Инициализация менеджера портфеля.
        
        Args:
            config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        self.config = config
        self.logger = logger or DRLLogger("portfolio_manager")
        
        # Состояние портфеля
        self.initial_balance = config.initial_balance
        self.balance_usdt = config.initial_balance
        self.balance_crypto = 0.0
        self.position_size = 0.0  # От -1 (полный шорт) до +1 (полный лонг)
        
        # История торговли
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[Dict[str, float]] = []
        
        # Метрики производительности
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_portfolio_value = config.initial_balance
        self.current_price = 0.0
        
        # Управление рисками
        self.max_drawdown_reached = 0.0
        self.consecutive_losses = 0
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        self.logger.info(f"Портфель инициализирован с балансом {config.initial_balance} USDT")
    
    def reset(self, initial_balance: Optional[float] = None):
        """Сброс портфеля к начальному состоянию."""
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.balance_usdt = self.initial_balance
        self.balance_crypto = 0.0
        self.position_size = 0.0
        
        self.trade_history.clear()
        self.portfolio_history.clear()
        
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_portfolio_value = self.initial_balance
        self.current_price = 0.0
        
        self.max_drawdown_reached = 0.0
        self.consecutive_losses = 0
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        self.logger.debug("Портфель сброшен")
    
    def execute_trade(
        self, 
        action: Union[int, np.ndarray, float], 
        price: float, 
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Выполнение торговой операции.
        
        Args:
            action: действие агента (continuous: -1 to 1, discrete: 0,1,2)
            price: цена исполнения
            timestamp: время сделки
            
        Returns:
            Информация о выполненной сделке
        """
        self.current_price = price
        
        # Интерпретация действия
        trade_action, target_position = self._interpret_action(action)
        
        if trade_action == "hold":
            return self._create_hold_trade_info(timestamp, price)
        
        # Расчет размера сделки
        trade_info = self._calculate_trade_size(trade_action, target_position, price)
        
        if trade_info["amount"] == 0:
            return self._create_hold_trade_info(timestamp, price)
        
        # Проверка рисков
        if not self._check_risk_limits(trade_info, price):
            self.logger.warning("Сделка отклонена из-за лимитов риска")
            return self._create_rejected_trade_info(timestamp, price, "Risk limits")
        
        # Выполнение сделки
        executed_trade = self._execute_validated_trade(trade_info, price, timestamp)
        
        # Обновление состояния
        self._update_portfolio_state(executed_trade)
        
        return executed_trade
    
    def _interpret_action(self, action: Union[int, np.ndarray, float]) -> tuple[str, float]:
        """Интерпретация действия агента."""
        if self.config.action_type == "continuous":
            # Continuous action: -1 (полный sell) до +1 (полный buy)
            if isinstance(action, np.ndarray):
                action = float(action[0])
            
            if abs(action) < 0.1:  # Мертвая зона для hold
                return "hold", self.position_size
            elif action > 0:
                return "buy", float(action)  # target position от 0 до 1
            else:
                return "sell", float(action)  # target position от -1 до 0
        
        else:  # discrete
            action_map = {0: "buy", 1: "sell", 2: "hold"}
            trade_action = action_map.get(int(action), "hold")
            
            if trade_action == "buy":
                return "buy", 0.5  # Покупка на 50% капитала
            elif trade_action == "sell":
                return "sell", -0.5 if self.position_size > 0 else 0  # Продажа половины позиции
            else:
                return "hold", self.position_size
    
    def _calculate_trade_size(self, trade_action: str, target_position: float, price: float) -> Dict[str, Any]:
        """Расчет размера сделки."""
        current_value = self.get_total_value()
        
        if trade_action == "buy":
            # Расчет для покупки
            target_crypto_value = current_value * abs(target_position)
            current_crypto_value = self.balance_crypto * price
            additional_crypto_value = target_crypto_value - current_crypto_value
            
            if additional_crypto_value <= 0:
                return {"action": "hold", "amount": 0, "value": 0}
            
            # Ограничение максимальным размером позиции
            max_position_value = current_value * self.config.max_position_size
            additional_crypto_value = min(additional_crypto_value, max_position_value - current_crypto_value)
            
            # Ограничение доступным балансом USDT
            available_usdt = self.balance_usdt * (1 - self.config.commission_rate)
            additional_crypto_value = min(additional_crypto_value, available_usdt)
            
            if additional_crypto_value < self.config.min_trade_amount:
                return {"action": "hold", "amount": 0, "value": 0}
            
            crypto_amount = additional_crypto_value / price
            commission = additional_crypto_value * self.config.commission_rate
            
            return {
                "action": "buy",
                "amount": crypto_amount,
                "value": additional_crypto_value,
                "commission": commission
            }
        
        elif trade_action == "sell":
            # Расчет для продажи
            if self.balance_crypto <= 0:
                return {"action": "hold", "amount": 0, "value": 0}
            
            if target_position >= 0:
                # Частичная продажа до целевой позиции
                target_crypto_value = current_value * target_position
                current_crypto_value = self.balance_crypto * price
                crypto_to_sell_value = current_crypto_value - target_crypto_value
            else:
                # Полная продажа
                crypto_to_sell_value = self.balance_crypto * price
            
            if crypto_to_sell_value <= 0:
                return {"action": "hold", "amount": 0, "value": 0}
            
            # Минимальный размер сделки
            if crypto_to_sell_value < self.config.min_trade_amount:
                return {"action": "hold", "amount": 0, "value": 0}
            
            crypto_amount = min(crypto_to_sell_value / price, self.balance_crypto)
            commission = crypto_to_sell_value * self.config.commission_rate
            
            return {
                "action": "sell", 
                "amount": crypto_amount,
                "value": crypto_to_sell_value,
                "commission": commission
            }
        
        return {"action": "hold", "amount": 0, "value": 0}
    
    def _check_risk_limits(self, trade_info: Dict[str, Any], price: float) -> bool:
        """Проверка лимитов риска с адаптивными настройками для обучения."""
        total_value = self.get_total_value()
        
        # Определяем режим (training vs production)
        training_mode = getattr(self.config, 'training_mode', True)
        
        if training_mode:
            # Мягкие лимиты для обучения DRL агентов
            max_risk_limit = 0.15  # 15% вместо стандартных 2%
            max_position_limit = 0.90  # 90% вместо 80%
            min_trade_limit = max(self.config.min_trade_amount, 10.0)  # Снижаем минимальную сделку
            max_drawdown_limit = 0.50  # 50% максимальная просадка для обучения
        else:
            # Строгие лимиты для продакшена
            max_risk_limit = self.config.max_risk_per_trade
            max_position_limit = self.config.max_position_size
            min_trade_limit = self.config.min_trade_amount
            max_drawdown_limit = self.config.max_drawdown_limit
        
        # Проверка максимального риска на сделку
        risk_amount = trade_info["value"] / total_value if total_value > 0 else 0
        if risk_amount > max_risk_limit:
            self.logger.debug(f"Сделка отклонена: риск {risk_amount:.3f} > лимит {max_risk_limit:.3f}")
            return False
        
        # Проверка максимального размера позиции
        if trade_info["action"] == "buy":
            future_crypto_value = (self.balance_crypto + trade_info["amount"]) * price
            future_position_ratio = future_crypto_value / total_value if total_value > 0 else 0
            if future_position_ratio > max_position_limit:
                self.logger.debug(f"Сделка отклонена: будущая позиция {future_position_ratio:.3f} > лимит {max_position_limit:.3f}")
                return False
        
        # Проверка минимального размера сделки
        if trade_info["value"] < min_trade_limit:
            self.logger.debug(f"Сделка отклонена: размер {trade_info['value']:.2f} < мин.лимит {min_trade_limit:.2f}")
            return False
        
        # Проверка доступного баланса
        if trade_info["action"] == "buy":
            required_usdt = trade_info["value"] + trade_info["commission"]
            if required_usdt > self.balance_usdt:
                self.logger.debug(f"Сделка отклонена: требуется ${required_usdt:.2f}, доступно ${self.balance_usdt:.2f}")
                return False
        elif trade_info["action"] == "sell":
            if trade_info["amount"] > self.balance_crypto:
                self.logger.debug(f"Сделка отклонена: продажа {trade_info['amount']:.6f}, доступно {self.balance_crypto:.6f}")
                return False
        
        # Проверка просадки
        current_drawdown = self.get_current_drawdown()
        if current_drawdown >= max_drawdown_limit:
            self.logger.debug(f"Сделка отклонена: просадка {current_drawdown:.3f} >= лимит {max_drawdown_limit:.3f}")
            return False
        
        return True
    
    def _execute_validated_trade(self, trade_info: Dict[str, Any], price: float, timestamp: str) -> Dict[str, Any]:
        """Выполнение проверенной сделки."""
        action = trade_info["action"]
        amount = trade_info["amount"]
        value = trade_info["value"]
        commission = trade_info["commission"]
        
        position_before = self.position_size
        
        if action == "buy":
            # Покупка криптовалюты
            self.balance_usdt -= (value + commission)
            self.balance_crypto += amount
            
            # Расчет P&L (для усреднения позиции)
            pnl = 0.0  # P&L рассчитывается при продаже
            
        elif action == "sell":
            # Продажа криптовалюты
            self.balance_usdt += (value - commission)
            self.balance_crypto -= amount
            
            # Расчет реализованной прибыли/убытка
            # Упрощенный расчет - можно улучшить до FIFO/LIFO
            avg_buy_price = self._get_average_buy_price()
            pnl = (price - avg_buy_price) * amount if avg_buy_price > 0 else 0.0
            self.realized_pnl += pnl
        
        # Обновление размера позиции
        total_value = self.get_total_value()
        if total_value > 0:
            self.position_size = (self.balance_crypto * price) / total_value
        
        # Создание записи о сделке
        trade = Trade(
            timestamp=timestamp,
            action=action,
            amount=amount,
            price=price,
            value=value,
            commission=commission,
            pnl=pnl,
            position_size_before=position_before,
            position_size_after=self.position_size
        )
        
        self.trade_history.append(trade)
        
        # Логирование
        self.logger.debug(f"Сделка выполнена: {action} {amount:.6f} @ ${price:.4f}, "
                         f"комиссия: ${commission:.4f}, P&L: ${pnl:.4f}")
        
        return {
            "executed": True,
            "action": action,
            "amount": amount,
            "price": price,
            "value": value,
            "commission": commission,
            "pnl": pnl,
            "position_size": self.position_size,
            "portfolio_value": total_value
        }
    
    def _create_hold_trade_info(self, timestamp: str, price: float) -> Dict[str, Any]:
        """Создание информации о hold действии."""
        return {
            "executed": False,
            "action": "hold",
            "amount": 0.0,
            "price": price,
            "value": 0.0,
            "commission": 0.0,
            "pnl": 0.0,
            "position_size": self.position_size,
            "portfolio_value": self.get_total_value()
        }
    
    def _create_rejected_trade_info(self, timestamp: str, price: float, reason: str) -> Dict[str, Any]:
        """Создание информации об отклоненной сделке."""
        return {
            "executed": False,
            "action": "rejected",
            "amount": 0.0,
            "price": price,
            "value": 0.0,
            "commission": 0.0,
            "pnl": 0.0,
            "position_size": self.position_size,
            "portfolio_value": self.get_total_value(),
            "reason": reason
        }
    
    def _update_portfolio_state(self, trade_info: Dict[str, Any]):
        """Обновление состояния портфеля после сделки."""
        # Обновление максимального значения портфеля
        current_value = self.get_total_value()
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        # Обновление истории портфеля
        self.portfolio_history.append({
            "timestamp": trade_info.get("timestamp", ""),
            "total_value": current_value,
            "balance_usdt": self.balance_usdt,
            "balance_crypto": self.balance_crypto,
            "position_size": self.position_size,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl()
        })
        
        # Обновление статистики проигрышных сделок
        if trade_info.get("pnl", 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def update_portfolio_value(self, current_price: float):
        """Обновление стоимости портфеля при изменении цены."""
        self.current_price = current_price
        self.unrealized_pnl = self._calculate_unrealized_pnl(current_price)
    
    def get_total_value(self) -> float:
        """Получение общей стоимости портфеля."""
        if self.current_price > 0:
            return self.balance_usdt + (self.balance_crypto * self.current_price)
        return self.balance_usdt
    
    def get_total_return(self) -> float:
        """Получение общей доходности портфеля."""
        if self.initial_balance == 0:
            return 0.0
        return (self.get_total_value() - self.initial_balance) / self.initial_balance
    
    def get_unrealized_pnl(self) -> float:
        """Получение нереализованной прибыли/убытка."""
        return self._calculate_unrealized_pnl(self.current_price)
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Расчет нереализованной прибыли/убытка."""
        if self.balance_crypto <= 0 or current_price <= 0:
            return 0.0
        
        avg_buy_price = self._get_average_buy_price()
        if avg_buy_price <= 0:
            return 0.0
        
        return (current_price - avg_buy_price) * self.balance_crypto
    
    def _get_average_buy_price(self) -> float:
        """Получение средней цены покупки."""
        if not self.trade_history:
            return 0.0
        
        total_bought = 0.0
        total_cost = 0.0
        
        for trade in self.trade_history:
            if trade.action == "buy":
                total_bought += trade.amount
                total_cost += trade.value
        
        return total_cost / total_bought if total_bought > 0 else 0.0
    
    def get_current_drawdown(self) -> float:
        """Получение текущей просадки."""
        if self.max_portfolio_value == 0:
            return 0.0
        
        current_value = self.get_total_value()
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # Обновление максимальной просадки
        self.max_drawdown_reached = max(self.max_drawdown_reached, drawdown)
        
        return drawdown
    
    def get_max_drawdown(self) -> float:
        """Получение максимальной просадки за период."""
        return self.max_drawdown_reached
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Получение метрик портфеля."""
        return {
            "total_value": self.get_total_value(),
            "total_return": self.get_total_return(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "current_drawdown": self.get_current_drawdown(),
            "max_drawdown": self.get_max_drawdown(),
            "position_size": self.position_size,
            "balance_usdt": self.balance_usdt,
            "balance_crypto": self.balance_crypto,
            "trades_count": len(self.trade_history),
            "consecutive_losses": self.consecutive_losses
        }
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Получение статистики торговли."""
        if not self.trade_history:
            return {"total_trades": 0}
        
        # Фильтрация реальных сделок (исключая hold)
        real_trades = [t for t in self.trade_history if t.action in ["buy", "sell"]]
        
        if not real_trades:
            return {"total_trades": 0}
        
        # Статистика по типам сделок
        buy_trades = [t for t in real_trades if t.action == "buy"]
        sell_trades = [t for t in real_trades if t.action == "sell"]
        
        # P&L статистика
        profitable_trades = [t for t in sell_trades if t.pnl > 0]
        losing_trades = [t for t in sell_trades if t.pnl < 0]
        
        return {
            "total_trades": len(real_trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "profitable_trades": len(profitable_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(profitable_trades) / len(sell_trades) if sell_trades else 0,
            "avg_profit": np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0,
            "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            "total_commission": sum(t.commission for t in real_trades),
            "largest_win": max([t.pnl for t in sell_trades]) if sell_trades else 0,
            "largest_loss": min([t.pnl for t in sell_trades]) if sell_trades else 0
        }