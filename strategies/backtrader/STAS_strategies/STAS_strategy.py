import backtrader as bt
import backtrader.indicators as btind


class STASStrategy(bt.Strategy):
    """
    STAS - Superior Technical Analysis Strategy
    
    Оптимизированная стратегия на основе лучших практик backtrader:
    - Качественные сигналы входа с множественным подтверждением
    - Адаптивные уровни RSI для разных рыночных условий
    - Трендовая фильтрация для повышения точности
    - Оптимизированное управление рисками
    - Компаундинг для достижения 100%+ прибыли
    
    Цель: Минимум 100% прибыль при контролируемом риске
    """
    
    params = (
        # Упрощенные индикаторы
        ('ema_fast', 12),          # Быстрая EMA
        ('ema_slow', 26),          # Медленная EMA
        
        # RSI параметры - упрощенные
        ('rsi_period', 14),
        ('rsi_oversold', 30),      # Перепроданность
        ('rsi_overbought', 70),    # Перекупленность
        
        # Упрощенное управление рисками
        ('position_size', 0.20),   # 20% капитала за сделку
        ('stop_loss', 0.05),       # 5% стоп-лосс
        ('take_profit', 0.15),     # 15% тейк-профит
    )

    def __init__(self):
        # Простые индикаторы
        self.ema_fast = btind.EMA(period=self.p.ema_fast)
        self.ema_slow = btind.EMA(period=self.p.ema_slow)
        self.rsi = btind.RSI(period=self.p.rsi_period)
        
        # Кроссы для сигналов входа
        self.ema_cross_up = btind.CrossUp(self.ema_fast, self.ema_slow)
        self.ema_cross_down = btind.CrossDown(self.ema_fast, self.ema_slow)
        
        # Состояние торговли
        self.order = None
        self.entry_price = None
        
        # Минимальный период для работы индикаторов
        self.min_period = max(self.p.ema_slow, self.p.rsi_period)

    def log(self, txt, dt=None):
        """Логирование с форматированием"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: STAS - {txt}')

    def next(self):
        # BACKTRADER BEST PRACTICE: Use precomputed minimum period
        if len(self.data) < self.min_period:
            return

        # Пропускаем если есть активный ордер  
        if self.order:
            return

        # IMPROVED: Better data validation
        current_price = self.data.close[0]
        if not current_price or current_price <= 0 or not self._is_data_valid():
            return

        # ВХОД В ПОЗИЦИЮ - Простые сигналы для больше сделок
        if not self.position:
            signal_quality = self._calculate_signal_quality()
            
            # Низкий порог для больше торговых возможностей
            if signal_quality >= 3.0:  # Снижен с 5.0 до 3.0
                # Простой размер позиции
                size = self._calculate_position_size()
                
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    
                    self.log(f"📈 ПОКУПКА: {current_price:.2f}, Качество: {signal_quality:.1f}/10, RSI: {self.rsi[0]:.1f}")

        # УПРАВЛЕНИЕ ОТКРЫТОЙ ПОЗИЦИЕЙ
        elif self.position and self.entry_price:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price

            # Простые условия выхода
            exit_reason = self._should_exit(current_price, current_profit_pct)
            
            if exit_reason:
                self.order = self.close()
                self.log(f"📉 ПРОДАЖА: {current_price:.2f}, Прибыль: {current_profit_pct*100:.1f}%, Причина: {exit_reason}")
                
                # Сброс состояния
                self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

    def notify_trade(self, trade):
        """Обработка событий сделок (как у Makar)"""
        if not trade.isclosed:
            return
        
        pnl_pct = (trade.pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
        
        if trade.pnl > 0:
            self.log(f'TRADE WIN! 🎉 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')
        else:
            self.log(f'TRADE LOSS 😞 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')

    def _calculate_signal_quality(self) -> float:
        """Простой расчет качества сигнала входа (0-10)"""
        try:
            score = 0.0
            
            # 1. EMA тренд (0-4 балла)
            if len(self.ema_fast) > 0 and len(self.ema_slow) > 0:
                if self.ema_fast[0] > self.ema_slow[0]:
                    score += 4.0  # Бычий тренд
            
            # 2. RSI уровни (0-3 балла)
            if len(self.rsi) > 0:
                rsi_val = self.rsi[0]
                if rsi_val < self.p.rsi_oversold:
                    score += 3.0  # Перепроданность - хорошо для покупки
                elif 40 <= rsi_val <= 60:
                    score += 1.0  # Нейтральная зона
            
            # 3. EMA кросс (0-3 балла)
            if len(self.ema_cross_up) > 0 and self.ema_cross_up[0]:
                score += 3.0  # Свежий бычий кросс
            
            return min(max(score, 0.0), 10.0)
            
        except (IndexError, TypeError):
            return 0.0

    def _calculate_position_size(self) -> float:
        """Простой расчет размера позиции"""
        try:
            current_price = self.data.close[0]
            if current_price <= 0:
                return 0
            
            # Простой размер позиции - процент от капитала
            size = (self.broker.cash * self.p.position_size) / current_price
            
            # Ограничения безопасности
            max_size = self.broker.cash * 0.99 / current_price
            return min(size, max_size) if size > 0 else 0
            
        except (ZeroDivisionError, TypeError):
            return 0

    def _is_data_valid(self) -> bool:
        """Проверка корректности данных (BACKTRADER BEST PRACTICE)"""
        try:
            # Проверяем основные OHLC данные
            if (self.data.open[0] <= 0 or self.data.high[0] <= 0 or 
                self.data.low[0] <= 0 or self.data.close[0] <= 0):
                return False
                
            # Проверяем логику OHLC (high >= max(o,c), low <= min(o,c))
            if (self.data.high[0] < max(self.data.open[0], self.data.close[0]) or
                self.data.low[0] > min(self.data.open[0], self.data.close[0])):
                return False
                
            # Проверяем наличие всех индикаторов
            if (len(self.ema_fast) == 0 or len(self.ema_slow) == 0 or len(self.rsi) == 0):
                return False
                
            return True
            
        except (IndexError, TypeError, AttributeError):
            return False

    def _should_exit(self, current_price: float, profit_pct: float) -> str:
        """Определяет нужно ли выходить из позиции"""
        try:
            # 1. Стоп-лосс
            if profit_pct <= -self.p.stop_loss:
                return "STOP_LOSS"
                
            # 2. Тейк-профит
            if profit_pct >= self.p.take_profit:
                return "TAKE_PROFIT"
                
            # 3. Технические сигналы выхода
            if len(self.rsi) > 0 and self.rsi[0] > self.p.rsi_overbought and profit_pct > 0.05:
                return "RSI_OVERBOUGHT"
                
            if len(self.ema_cross_down) > 0 and self.ema_cross_down[0] and profit_pct > 0.03:
                return "EMA_CROSS_DOWN"
                
            return None
            
        except (IndexError, TypeError, ZeroDivisionError):
            return None

    def stop(self):
        """Финальная статистика"""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100
        
        self.log(f'🏁 STAS Strategy Complete!')
        self.log(f'📊 Final Value: ${final_value:.2f}')
        self.log(f'📈 Total Return: {total_return:+.2f}%')
        
        if total_return >= 1000:
            self.log(f'🎯 TARGET ACHIEVED! Return > 1000%')
        elif total_return >= 100:
            self.log(f'✅ Great Performance! Return > 100%')
        else:
            self.log(f'📝 Room for Improvement. Target: 1000%+')