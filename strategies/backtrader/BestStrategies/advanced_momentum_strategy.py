import backtrader as bt
import backtrader.indicators as btind


class AdvancedMomentumStrategy(bt.Strategy):
    """
    Продвинутая моментум стратегия с множественными фильтрами
    
    Использует комбинацию технических индикаторов:
    - RSI для определения перекупленности/перепроданности
    - EMA пересечения для тренда
    - MACD для подтверждения моментума
    - Bollinger Bands для волатильности
    """
    
    params = (
        # RSI параметры
        ('rsi_period', 14),
        ('rsi_oversold', 25),
        ('rsi_overbought', 75),
        
        # EMA параметры  
        ('ema_fast', 12),
        ('ema_slow', 26),
        
        # MACD параметры
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # Bollinger Bands параметры
        ('bb_period', 20),
        ('bb_dev', 2.0),
        
        # Управление рисками
        ('position_size', 0.95),
        ('stop_loss', 0.08),
        ('take_profit', 0.25),
        ('trailing_stop', 0.15),
        
        # Фильтры
        ('min_volume_filter', True),
        ('trend_filter', True),
    )

    def __init__(self):
        # Основные индикаторы
        self.rsi = btind.RSI(period=self.p.rsi_period)
        self.ema_fast = btind.EMA(period=self.p.ema_fast)
        self.ema_slow = btind.EMA(period=self.p.ema_slow)
        
        self.macd = btind.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        self.bb = btind.BollingerBands(
            period=self.p.bb_period,
            devfactor=self.p.bb_dev
        )
        
        # Производные сигналы
        self.ema_cross = btind.CrossOver(self.ema_fast, self.ema_slow)
        self.macd_cross = btind.CrossOver(self.macd.macd, self.macd.signal)
        self.trend_up = self.ema_fast > self.ema_slow
        
        # Состояние
        self.order = None
        self.entry_price = None
        self.highest_price = None
        
        # Статистика
        self.signal_strength = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: ADV-MOMENTUM - {txt}')

    def next(self):
        # Ждем достаточно данных
        if len(self.data) < max(self.p.ema_slow, self.p.bb_period, self.p.rsi_period):
            return
            
        if self.order:
            return

        current_price = self.data.close[0]
        volume = self.data.volume[0]

        # ВХОД В ПОЗИЦИЮ
        if not self.position:
            self.signal_strength = self._calculate_signal_strength()
            
            # Упрощенные условия для LONG (больше сделок)
            long_conditions = [
                self.rsi[0] < self.p.rsi_oversold,  # RSI перепродан
                self.ema_cross[0] > 0,  # Бычий кросс EMA
                self.macd_cross[0] > 0,  # Бычий кросс MACD
                current_price <= self.bb.lines.bot[0] * 1.02,  # Близко к нижней полосе (с запасом)
                self.macd.macd[0] > self.macd.macd[-1],  # MACD растет
                current_price > self.data.close[-1],  # Цена растет
                self.rsi[0] < 40,  # Дополнительный RSI фильтр
                self.trend_up[0],  # Тренд вверх
            ]
            
            # Фильтры (более мягкие)
            volume_ok = True  # Убираем фильтр объема для больше сделок
            if self.p.min_volume_filter and hasattr(self.data, 'volume') and len(self.data.volume) > 10:
                avg_vol = sum(self.data.volume.get(ago=i, size=1) for i in range(1, 11)) / 10
                volume_ok = volume > avg_vol * 0.5  # Снижаем требования к объему
            
            # СНИЖАЕМ требования: минимум 2 условия вместо 3
            if sum(long_conditions) >= 2 and volume_ok:
                size = (self.broker.cash * self.p.position_size) / current_price
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.highest_price = current_price
                    
                    self.log(f'BUY SIGNAL! Price: {current_price:.2f}, '
                            f'Signal Strength: {self.signal_strength:.2f}, '
                            f'RSI: {self.rsi[0]:.1f}, Conditions: {sum(long_conditions)}/8')

        # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
        elif self.position and self.entry_price:
            # Обновляем максимальную цену
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            profit_pct = (current_price - self.entry_price) / self.entry_price
            drawdown_from_peak = (self.highest_price - current_price) / self.highest_price
            
            # Условия выхода
            exit_conditions = [
                # Фиксированный стоп-лосс
                profit_pct < -self.p.stop_loss,
                
                # Фиксированный тейк-профит
                profit_pct > self.p.take_profit,
                
                # Трейлинг стоп
                drawdown_from_peak > self.p.trailing_stop,
                
                # Технические сигналы
                self.rsi[0] > self.p.rsi_overbought and profit_pct > 0.02,
                self.ema_cross[0] < 0,  # Медвежий кросс EMA
                self.macd_cross[0] < 0,  # Медвежий кросс MACD
                current_price >= self.bb.lines.top[0] and profit_pct > 0.05,
            ]
            
            if any(exit_conditions):
                self.order = self.close()
                
                # Определяем причину выхода
                if profit_pct < -self.p.stop_loss:
                    reason = "STOP LOSS"
                elif profit_pct > self.p.take_profit:
                    reason = "TAKE PROFIT"
                elif drawdown_from_peak > self.p.trailing_stop:
                    reason = "TRAILING STOP"
                else:
                    reason = "TECHNICAL SIGNAL"
                
                self.log(f'SELL! {reason} - Price: {current_price:.2f}, '
                        f'Profit: {profit_pct*100:.2f}%, Max Price: {self.highest_price:.2f}')
                
                self.entry_price = None
                self.highest_price = None

    def _calculate_signal_strength(self) -> float:
        """Расчет силы сигнала от 0 до 10"""
        score = 0
        
        # RSI score (0-2)
        if self.rsi[0] < 20:
            score += 2
        elif self.rsi[0] < 30:
            score += 1.5
        elif self.rsi[0] < 40:
            score += 1
            
        # Trend score (0-2)
        if self.trend_up[0]:
            score += 1
            if self.ema_fast[0] > self.ema_fast[-1]:  # EMA растет
                score += 1
                
        # MACD score (0-2)
        if self.macd.macd[0] > self.macd.signal[0]:
            score += 1
            if self.macd.macd[0] > self.macd.macd[-1]:
                score += 1
                
        # Bollinger score (0-2)
        bb_position = (self.data.close[0] - self.bb.lines.bot[0]) / (self.bb.lines.top[0] - self.bb.lines.bot[0])
        if bb_position < 0.2:  # Близко к нижней полосе
            score += 2
        elif bb_position < 0.4:
            score += 1
            
        # Volume score (0-2)
        if hasattr(self.data, 'volume') and len(self.data.volume) > 5:
            recent_volume = self.data.volume[0]
            avg_volume = sum(self.data.volume.get(ago=i, size=1) for i in range(1, 6)) / 5
            if recent_volume > avg_volume * 1.5:
                score += 2
            elif recent_volume > avg_volume * 1.2:
                score += 1
        
        return min(score, 10)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.6f}, Cost: ${order.executed.value:.2f}')
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.6f}, Value: ${order.executed.value:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        pnl_pct = (trade.pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
        
        if trade.pnl > 0:
            self.log(f'TRADE WIN! 🎉 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')
        else:
            self.log(f'TRADE LOSS 😞 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')

    def stop(self):
        final_value = self.broker.getvalue()
        self.log(f'Advanced Momentum Strategy finished! Final Value: ${final_value:.2f}')


class AdaptiveTrendFollowingStrategy(bt.Strategy):
    """
    Адаптивная трендследящая стратегия с динамическими параметрами
    
    Особенности:
    - Автоматическая адаптация к волатильности
    - Динамические уровни стоп-лосса и тейк-профита
    - Множественные таймфреймы для подтверждения
    """
    
    params = (
        # Основные параметры
        ('atr_period', 14),
        ('trend_period', 20),
        ('position_size', 0.90),
        
        # Адаптивные множители
        ('sl_atr_mult', 2.0),
        ('tp_atr_mult', 4.0),
        ('vol_lookback', 20),
        
        # Фильтры тренда
        ('min_trend_strength', 0.6),
        ('volume_filter', True),
    )

    def __init__(self):
        # Основные индикаторы
        self.atr = btind.ATR(period=self.p.atr_period)
        self.sma_trend = btind.SMA(period=self.p.trend_period)
        self.ema_short = btind.EMA(period=8)
        self.ema_long = btind.EMA(period=21)
        
        # Волатильность и адаптация
        self.stddev = btind.StandardDeviation(period=self.p.vol_lookback)
        
        # Трендовые сигналы
        self.trend_up = self.ema_short > self.ema_long
        self.price_above_trend = self.data.close > self.sma_trend
        
        # Состояние
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_bars = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: ADAPTIVE-TREND - {txt}')

    def next(self):
        if len(self.data) < max(self.p.atr_period, self.p.trend_period, self.p.vol_lookback):
            return
            
        if self.order:
            return

        current_price = self.data.close[0]
        current_atr = self.atr[0]
        
        # ВХОД В ПОЗИЦИЮ  
        if not self.position:
            trend_strength = self._calculate_trend_strength()
            
            # Условия для входа
            entry_conditions = [
                self.trend_up[0],  # Краткосрочный тренд вверх
                self.price_above_trend[0],  # Цена выше долгосрочного тренда
                trend_strength > self.p.min_trend_strength,  # Достаточная сила тренда
                current_price > self.data.close[-1],  # Цена растет
                self.ema_short[0] > self.ema_short[-1],  # Краткосрочная EMA растет
            ]
            
            # Фильтр объема
            volume_ok = True
            if self.p.volume_filter and hasattr(self.data, 'volume'):
                avg_volume = sum(self.data.volume.get(ago=i, size=1) for i in range(1, 11)) / 10
                volume_ok = self.data.volume[0] > avg_volume * 0.8
            
            if sum(entry_conditions) >= 4 and volume_ok:
                # Адаптивный размер позиции на основе волатильности
                volatility_adj = max(0.5, 1 - (self.stddev[0] / current_price))
                adjusted_size = self.p.position_size * volatility_adj
                
                size = (self.broker.cash * adjusted_size) / current_price
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    
                    # Адаптивные уровни на основе ATR
                    self.stop_price = current_price - (current_atr * self.p.sl_atr_mult)
                    self.target_price = current_price + (current_atr * self.p.tp_atr_mult)
                    self.position_bars = 0
                    
                    self.log(f'BUY! Price: {current_price:.2f}, ATR: {current_atr:.2f}, '
                            f'Stop: {self.stop_price:.2f}, Target: {self.target_price:.2f}, '
                            f'Trend Strength: {trend_strength:.2f}')

        # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
        elif self.position:
            self.position_bars += 1
            
            # Динамическое обновление стопа (трейлинг)
            if current_price > self.entry_price:
                new_stop = current_price - (current_atr * self.p.sl_atr_mult)
                self.stop_price = max(self.stop_price, new_stop)
            
            # Условия выхода
            exit_signal = (
                current_price <= self.stop_price or  # Стоп-лосс
                current_price >= self.target_price or  # Тейк-профит
                not self.trend_up[0] or  # Тренд развернулся
                self.position_bars >= 50  # Максимальное время в позиции
            )
            
            if exit_signal:
                self.order = self.close()
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                
                self.log(f'SELL! Price: {current_price:.2f}, '
                        f'Profit: {profit_pct:+.2f}%, Bars: {self.position_bars}')

    def _calculate_trend_strength(self) -> float:
        """Расчет силы тренда от 0 до 1"""
        if len(self.data) < 20:
            return 0
            
        # Анализ последних 20 баров
        closes = [self.data.close[-i] for i in range(20)]
        
        # Подсчет растущих баров
        up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        
        # Сила тренда = процент растущих баров
        trend_strength = up_bars / (len(closes) - 1)
        
        # Дополнительный фактор: положение цены относительно SMA
        price_position = (self.data.close[0] - self.sma_trend[0]) / self.sma_trend[0]
        price_factor = min(1, max(0, price_position * 10 + 0.5))
        
        return (trend_strength + price_factor) / 2

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE RESULT: PnL: ${trade.pnl:.2f}')