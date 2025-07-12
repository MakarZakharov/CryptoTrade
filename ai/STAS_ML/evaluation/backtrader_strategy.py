"""
Backtrader стратегія для ML моделей STAS_ML.
"""

import backtrader as bt
import numpy as np
from typing import Optional, Any, Dict


class MLPredictionStrategy(bt.Strategy):
    """
    Стратегія торгівлі на основі предсказаний ML моделі.
    
    Використовує заздалегідь обчислені предсказання ML моделі
    для прийняття торгових рішень з розширеним ризик-менеджментом.
    """
    
    params = dict(
        # Основні параметри
        position_size=0.15,           # Розмір позиції (15% від капіталу)
        
        # Ризик-менеджмент
        stop_loss_pct=0.01,          # Стоп-лосс 1%
        take_profit_pct=0.15,        # Тейк-профіт 15%
        trailing_stop_pct=0.03,      # Трейлінг стоп 3%
        max_drawdown_limit=0.06,     # Максимальна просадка 6%
        
        # Фільтрація сигналів
        confidence_threshold=0.85,    # Мінімальна впевненість 85%
        min_prediction_strength=0.5,  # Мінімальна сила сигналу
        
        # Додаткові параметри
        printlog=True,               # Виводити логи
        debug_mode=False,            # Режим відладки
    )
    
    def __init__(self):
        """Ініціалізація стратегії."""
        # Основні змінні
        self.predictions = None
        self.prediction_probabilities = None
        self.current_prediction_index = 0
        
        # Стан торгівлі
        self.order = None
        self.position_entry_price = 0
        self.position_peak_profit = 0
        self.consecutive_losses = 0
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.max_balance = self.broker.getcash()
        
        # Захист від просадки
        self.initial_cash = self.broker.getcash()
        
    def set_predictions(self, predictions: np.ndarray, 
                       prediction_probabilities: Optional[np.ndarray] = None):
        """
        Встановити предсказання ML моделі.
        
        Args:
            predictions: Масив предсказань (0/1 для direction)
            prediction_probabilities: Вірогідності предсказань (опціонально)
        """
        self.predictions = predictions
        self.prediction_probabilities = prediction_probabilities
        self.current_prediction_index = 0
        
        if self.params.debug_mode:
            self.log(f"Завантажено {len(predictions)} предсказань")
    
    def log(self, txt, dt=None, doprint=None):
        """Функція логування."""
        if doprint is None:
            doprint = self.params.printlog
            
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """Обробка статусу ордерів."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'ПОКУПКА: ${order.executed.price:.2f}, '
                        f'Розмір: {order.executed.size}, '
                        f'Комісія: ${order.executed.comm:.2f}')
                self.position_entry_price = order.executed.price
                self.position_peak_profit = 0
                
            elif order.issell():
                self.log(f'ПРОДАЖ: ${order.executed.price:.2f}, '
                        f'Розмір: {order.executed.size}, '
                        f'Комісія: ${order.executed.comm:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ОРДЕР ВІДХИЛЕНО: {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Обробка закритих угод."""
        if not trade.isclosed:
            return
        
        self.total_trades += 1
        
        if trade.pnlcomm > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            result = "ПРИБУТОК"
        else:
            self.consecutive_losses += 1
            result = "ЗБИТОК"
        
        win_rate = (self.winning_trades / self.total_trades) * 100
        
        self.log(f'{result}: ${trade.pnlcomm:.2f} | '
                f'Угод: {self.total_trades} | '
                f'Винрейт: {win_rate:.1f}%')
    
    def get_current_prediction(self) -> tuple:
        """
        Отримати поточне предсказання та впевненість.
        
        Returns:
            tuple: (prediction, confidence)
        """
        if (self.predictions is None or 
            self.current_prediction_index >= len(self.predictions)):
            return None, 0.0
        
        prediction = self.predictions[self.current_prediction_index]
        
        # Розрахунок впевненості
        if self.prediction_probabilities is not None:
            if self.prediction_probabilities.ndim == 2:
                # Для бінарної класифікації [prob_class_0, prob_class_1]
                confidence = np.max(self.prediction_probabilities[self.current_prediction_index])
            else:
                # Для простого масиву вірогідностей
                confidence = self.prediction_probabilities[self.current_prediction_index]
        else:
            # Якщо вірогідності не надані, використовуємо фіксовану впевненість
            confidence = 0.9
        
        return prediction, confidence
    
    def check_risk_management(self) -> bool:
        """
        Перевірка ризик-менеджменту.
        
        Returns:
            bool: True якщо торгівля дозволена
        """
        current_value = self.broker.getvalue()
        
        # Оновлюємо максимальний баланс
        if current_value > self.max_balance:
            self.max_balance = current_value
        
        # Перевірка максимальної просадки
        current_drawdown = (self.max_balance - current_value) / self.max_balance
        if current_drawdown > self.params.max_drawdown_limit:
            if self.params.debug_mode:
                self.log(f"ЗУПИНКА: Просадка {current_drawdown:.1%} > {self.params.max_drawdown_limit:.1%}")
            return False
        
        # Перевірка серії збитків
        if self.consecutive_losses > 5:
            if self.params.debug_mode:
                self.log(f"ОБЕРЕЖНІСТЬ: {self.consecutive_losses} збитків підряд")
            return True  # Продовжуємо, але обережно
        
        return True
    
    def manage_position(self) -> bool:
        """
        Управління існуючою позицією.
        
        Returns:
            bool: True якщо позицію закрито
        """
        if not self.position:
            return False
        
        current_price = self.data.close[0]
        
        if self.position.size > 0:  # Лонг позиція
            profit_pct = (current_price - self.position_entry_price) / self.position_entry_price
            
            # Оновлюємо пік прибутку
            if profit_pct > self.position_peak_profit:
                self.position_peak_profit = profit_pct
            
            # Тейк-профіт
            if profit_pct >= self.params.take_profit_pct:
                self.log(f"ТЕЙК-ПРОФІТ: {profit_pct:.1%}")
                self.order = self.sell(size=self.position.size)
                return True
            
            # Трейлінг стоп
            elif (self.position_peak_profit > self.params.trailing_stop_pct and 
                  (self.position_peak_profit - profit_pct) > self.params.trailing_stop_pct):
                self.log(f"ТРЕЙЛІНГ СТОП: Пік {self.position_peak_profit:.1%}, Поточний {profit_pct:.1%}")
                self.order = self.sell(size=self.position.size)
                return True
            
            # Стоп-лосс
            elif profit_pct <= -self.params.stop_loss_pct:
                self.log(f"СТОП-ЛОСС: {profit_pct:.1%}")
                self.order = self.sell(size=self.position.size)
                return True
        
        elif self.position.size < 0:  # Шорт позиція
            profit_pct = (self.position_entry_price - current_price) / self.position_entry_price
            
            # Оновлюємо пік прибутку
            if profit_pct > self.position_peak_profit:
                self.position_peak_profit = profit_pct
            
            # Тейк-профіт для шорту
            if profit_pct >= self.params.take_profit_pct:
                self.log(f"ТЕЙК-ПРОФІТ (ШОРТ): {profit_pct:.1%}")
                self.order = self.buy(size=abs(self.position.size))
                return True
            
            # Трейлінг стоп для шорту
            elif (self.position_peak_profit > self.params.trailing_stop_pct and 
                  (self.position_peak_profit - profit_pct) > self.params.trailing_stop_pct):
                self.log(f"ТРЕЙЛІНГ СТОП (ШОРТ): Пік {self.position_peak_profit:.1%}, Поточний {profit_pct:.1%}")
                self.order = self.buy(size=abs(self.position.size))
                return True
            
            # Стоп-лосс для шорту
            elif profit_pct <= -self.params.stop_loss_pct:
                self.log(f"СТОП-ЛОСС (ШОРТ): {profit_pct:.1%}")
                self.order = self.buy(size=abs(self.position.size))
                return True
        
        return False
    
    def calculate_position_size(self) -> int:
        """
        Розрахувати розмір позиції з урахуванням ризик-менеджменту.
        
        Returns:
            int: Розмір позиції в акціях
        """
        cash = self.broker.getcash()
        price = self.data.close[0]
        
        # Базовий розмір позиції
        base_size = int((cash * self.params.position_size) / price)
        
        # Адаптивне зменшення при серії збитків
        if self.consecutive_losses > 3:
            reduction_factor = 0.5 ** (self.consecutive_losses - 3)
            base_size = int(base_size * reduction_factor)
        
        # Мінімальний розмір позиції
        min_size = int(100 / price)  # Мінімум $100
        
        return max(base_size, min_size)
    
    def next(self):
        """Основна логіка стратегії на кожному кроці."""
        # Перевіряємо чи є активний ордер
        if self.order:
            return
        
        # Перевіряємо ризик-менеджмент
        if not self.check_risk_management():
            return
        
        # Управляємо існуючою позицією
        if self.manage_position():
            return
        
        # Отримуємо поточне предсказання
        prediction, confidence = self.get_current_prediction()
        
        if prediction is None:
            return
        
        # Перевіряємо впевненість
        if confidence < self.params.confidence_threshold:
            if self.params.debug_mode:
                self.log(f"СЛАБКИЙ СИГНАЛ: Впевненість {confidence:.1%} < {self.params.confidence_threshold:.1%}")
            self.current_prediction_index += 1
            return
        
        current_price = self.data.close[0]
        
        # Логіка торгівлі
        if prediction == 1 and not self.position:  # Сигнал покупки
            size = self.calculate_position_size()
            if size > 0:
                self.log(f"СИГНАЛ ПОКУПКИ: Впевненість {confidence:.1%}, Розмір: {size}")
                self.order = self.buy(size=size)
                
        elif prediction == 0 and not self.position:  # Сигнал продажу (шорт)
            size = self.calculate_position_size()
            if size > 0:
                self.log(f"СИГНАЛ ПРОДАЖУ: Впевненість {confidence:.1%}, Розмір: {size}")
                self.order = self.sell(size=size)
        
        # Переходимо до наступного предсказання
        self.current_prediction_index += 1
    
    def stop(self):
        """Викликається в кінці бектесту."""
        final_value = self.broker.getvalue()
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        self.log(f"=== ПІДСУМКИ ML СТРАТЕГІЇ ===", doprint=True)
        self.log(f"Початковий капітал: ${self.initial_cash:,.2f}", doprint=True)
        self.log(f"Фінальний капітал: ${final_value:,.2f}", doprint=True)
        self.log(f"Загальна доходність: {total_return:+.2f}%", doprint=True)
        self.log(f"Всього угод: {self.total_trades}", doprint=True)
        self.log(f"Виграшних угод: {self.winning_trades}", doprint=True)
        self.log(f"Винрейт: {win_rate:.1f}%", doprint=True)
        self.log(f"Послідовні збитки: {self.consecutive_losses}", doprint=True)
        self.log("=" * 35, doprint=True)


class MLPredictionDataFeed(bt.feeds.PandasData):
    """
    Спеціальний DataFeed для ML предсказань.
    
    Розширює стандартний PandasData додатковими полями
    для предсказань та вірогідностей.
    """
    
    lines = ('prediction', 'confidence')
    params = (
        ('prediction', -1),  # Колонка з предсказаннями
        ('confidence', -1),  # Колонка з впевненістю
    )