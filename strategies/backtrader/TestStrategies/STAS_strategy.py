import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import math


class STASStrategy(bt.Strategy):
    """
    STAS - Superior Technical Analysis Strategy V2.0
    
    Продвинутая стратегия с компаундингом для достижения 500%+ прибыли:
    - Динамическое реинвестирование прибыли через order_target_percent
    - Адаптивное управление капиталом с учетом роста портфеля
    - Оптимизированные параметры для максимального роста
    - Интеллектуальное масштабирование позиций
    - Компаундинг + качественные сигналы = ЭКСПОНЕНЦИАЛЬНЫЙ РОСТ
    
    Цель: Минимум 500% прибыль через компаундинг
    """
    
    params = (
        # Основные индикаторы (оптимизированы для 15м таймфрейма)
        ('ema_fast', 8),           # Быстрая EMA для краткосрочного тренда
        ('ema_slow', 21),          # Медленная EMA для среднесрочного тренда  
        ('ema_trend', 50),         # Долгосрочная EMA для фильтра тренда
        
        # RSI параметры (адаптивные уровни)
        ('rsi_period', 14),
        ('rsi_oversold_strong', 25), # Сильная перепроданность  
        ('rsi_oversold', 35),        # Обычная перепроданность
        ('rsi_overbought', 65),      # Обычная перекупленность
        ('rsi_overbought_strong', 75), # Сильная перекупленность
        
        # MACD параметры (оптимизированы)
        ('macd_fast', 12),
        ('macd_slow', 26), 
        ('macd_signal', 9),
        
        # СБАЛАНСИРОВАННОЕ управление рисками для устойчивого роста
        ('base_position_percent', 0.35), # Консервативный базовый % капитала  
        ('max_position_percent', 0.60),  # Максимальный % для лучших сигналов
        ('stop_loss', 0.04),            # 4% стоп-лосс (жесткий контроль потерь)
        ('take_profit', 0.12),          # 12% тейк-профит (реалистичные цели)
        ('trailing_stop', 0.08),        # Трейлинг стоп при 8% прибыли
        ('trailing_dist', 0.03),        # Расстояние трейлинга 3%
        ('max_dd_threshold', 0.15),     # Жесткий контроль просадки 15%
        ('emergency_dd_threshold', 0.25), # Аварийная просадка 25%
        ('dd_position_reduction', 0.3), # Агрессивное снижение при просадке
        
        # Kelly Criterion и динамическое позиционирование (КОНСЕРВАТИВНО)
        ('use_kelly_criterion', True),   # Использовать Kelly для размера позиций
        ('kelly_lookback', 50),         # Увеличенный период для стабильности
        ('max_kelly_fraction', 0.25),   # Консервативная Kelly доля (25% для контроля риска)
        ('volatility_lookback', 30),    # Увеличенный период волатильности
        ('vol_target', 0.02),          # Пониженная целевая волатильность (2%)
        
        # СТРОГИЕ фильтры для качественных сигналов
        ('volume_filter', True),
        ('trend_strength_min', 0.6),    # Повышен для качественных сигналов
        ('signal_quality_min', 5.0),    # Значительно повышен порог качества
        ('max_risk_per_trade', 0.02),   # Снижен риск на сделку до 2%
        ('max_portfolio_heat', 0.10),   # Снижена "нагретость" портфеля 10%
        
        # Режимы рынка и адаптация (ОПТИМИЗИРОВАНО ДЛЯ CRYPTO)
        ('market_regime_period', 30),   # Быстрая адаптация к режиму рынка
        ('trending_threshold', 0.4),    # Снижен порог для трендового рынка
        ('ranging_reduction', 0.7),     # Меньше снижение в боковике
    )

    def __init__(self):
        # Основные индикаторы для качественного анализа
        self.ema_fast = btind.EMA(period=self.p.ema_fast)
        self.ema_slow = btind.EMA(period=self.p.ema_slow)
        self.ema_trend = btind.EMA(period=self.p.ema_trend)
        
        # RSI с адаптивными уровнями
        self.rsi = btind.RSI(period=self.p.rsi_period)
        
        # MACD для подтверждения моментума
        self.macd = btind.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # Продвинутые индикаторы для риск-менеджмента
        self.atr = btind.ATR(period=14)  # Для адаптивных стопов
        self.volume_sma = btind.SMA(self.data.volume, period=20) if hasattr(self.data, 'volume') else None
        
        # Волатильность для Kelly Criterion и позиционирования
        self.returns = btind.PctChange(self.data.close, period=1)
        self.volatility = btind.StdDev(self.returns, period=self.p.volatility_lookback)
        
        # Индикаторы режима рынка
        self.market_regime_sma = btind.SMA(period=self.p.market_regime_period)
        self.price_vs_sma = self.data.close / self.market_regime_sma
        
        # Bollinger Bands для определения экстремальных условий
        self.bb = btind.BollingerBands(period=20, devfactor=2.0)
        
        # Кроссы для точных сигналов входа
        self.ema_cross_up = btind.CrossUp(self.ema_fast, self.ema_slow)
        self.ema_cross_down = btind.CrossDown(self.ema_fast, self.ema_slow)
        self.macd_cross_up = btind.CrossUp(self.macd.macd, self.macd.signal)
        self.macd_cross_down = btind.CrossDown(self.macd.macd, self.macd.signal)
        
        # Состояние торговли
        self.order = None
        self.entry_price = None
        self.trailing_stop_price = None
        self.highest_price = None
        
        # Продвинутое управление рисками и Kelly Criterion
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_seen = 0.0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Kelly Criterion отслеживание
        self.trade_history = []  # История сделок для Kelly
        self.kelly_fraction = 0.0
        self.current_win_rate = 0.0
        self.current_avg_win_loss_ratio = 1.0
        
        # Управление портфельными рисками
        self.portfolio_heat = 0.0  # Текущая "нагретость" портфеля
        self.last_entry_time = None
        self.market_regime = "neutral"  # trending, ranging, neutral
        
        # Экстренный режим при критической просадке
        self.emergency_mode = False
        self.trades_since_emergency = 0

    def log(self, txt, dt=None):
        """Логирование с форматированием"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: STAS - {txt}')

    def _calculate_kelly_fraction(self):
        """Расчет Kelly Criterion на основе истории сделок"""
        if len(self.trade_history) < 10:  # Минимум 10 сделок для расчета
            return self.p.max_kelly_fraction * 0.5  # Консервативный подход
        
        recent_trades = self.trade_history[-self.p.kelly_lookback:]
        if not recent_trades:
            return 0.0
        
        wins = [t for t in recent_trades if t > 0]
        losses = [abs(t) for t in recent_trades if t < 0]
        
        if not wins or not losses:
            return self.p.max_kelly_fraction * 0.3  # Очень консервативно
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly Formula: f = (bp - q) / b
        # где b = win_loss_ratio, p = win_rate, q = 1 - win_rate
        kelly = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        
        # Ограничиваем Kelly для безопасности
        kelly = max(0.0, min(kelly, self.p.max_kelly_fraction))
        
        # Дополнительная консервативная корректировка на основе волатильности
        if len(self.volatility) > 0 and self.volatility[0] > 0:
            vol_adjustment = min(1.0, self.p.vol_target / self.volatility[0])
            kelly *= vol_adjustment
        
        self.kelly_fraction = kelly
        self.current_win_rate = win_rate
        self.current_avg_win_loss_ratio = win_loss_ratio
        
        return kelly

    def _detect_market_regime(self):
        """Определение режима рынка: trending, ranging, neutral"""
        if len(self.price_vs_sma) < 20:
            return "neutral"
        
        # Анализ тренда через отношение цены к SMA
        recent_ratio = [self.price_vs_sma[-i] for i in range(1, 21)]  # Последние 20 периодов
        
        # Считаем, сколько периодов цена была выше/ниже SMA
        above_sma = sum(1 for r in recent_ratio if r > 1.02)  # 2% буфер
        below_sma = sum(1 for r in recent_ratio if r < 0.98)  # 2% буфер
        
        if above_sma >= int(20 * self.p.trending_threshold):
            return "trending_up"
        elif below_sma >= int(20 * self.p.trending_threshold):
            return "trending_down"
        else:
            return "ranging"

    def _calculate_portfolio_heat(self):
        """Расчет текущей 'нагретости' портфеля (экспозиции к риску)"""
        if not self.position:
            return 0.0
        
        current_value = self.broker.getvalue()
        position_value = abs(self.position.value)
        
        # Базовая нагретость = размер позиции / портфель
        base_heat = position_value / current_value if current_value > 0 else 0.0
        
        # Корректировка на волатильность
        if len(self.volatility) > 0 and self.volatility[0] > 0:
            vol_multiplier = self.volatility[0] / self.p.vol_target
            base_heat *= vol_multiplier
        
        return min(base_heat, 1.0)

    def _should_reduce_position_for_risk(self):
        """СТРОГАЯ проверка необходимости снижения размера позиции из-за рисков"""
        # Критическая просадка - ЭКСТРЕННАЯ остановка
        if self.current_drawdown >= self.p.emergency_dd_threshold:
            self.emergency_mode = True
            return 0.05  # Только 5% капитала в экстренном режиме
        
        # Любая просадка выше 10% - серьезное снижение
        if self.current_drawdown >= self.p.max_dd_threshold * 0.7:
            return 0.2  # Только 20% от нормального размера
        elif self.current_drawdown >= self.p.max_dd_threshold * 0.5:
            return 0.4  # 40% от нормального размера
        elif self.current_drawdown >= self.p.max_dd_threshold * 0.3:
            return 0.6  # 60% от нормального размера
        
        # Последовательные убытки - АГРЕССИВНОЕ снижение
        if self.consecutive_losses >= 3:
            return 0.1  # Минимальные позиции
        elif self.consecutive_losses >= 2:
            return 0.3  # Сильно сниженные позиции
        elif self.consecutive_losses >= 1:
            return 0.6  # Умеренно сниженные позиции
        
        # Высокая волатильность - осторожность
        if len(self.volatility) > 0:
            if self.volatility[0] > self.p.vol_target * 3:
                return 0.2  # Минимальные позиции при экстремальной волатильности
            elif self.volatility[0] > self.p.vol_target * 2:
                return 0.4  # Сниженные позиции при высокой волатильности
        
        # Боковой рынок - значительное снижение
        if self.market_regime == "ranging":
            return 0.3  # Только 30% в боковике
        
        # Если все в порядке, но общий риск высок
        portfolio_risk = self.portfolio_heat
        if portfolio_risk > self.p.max_portfolio_heat * 0.8:
            return 0.5
        
        return 1.0  # Полная позиция только при идеальных условиях

    def _calculate_adaptive_stops(self):
        """Расчет адаптивных стоп-лоссов на основе ATR"""
        if len(self.atr) == 0:
            return self.p.stop_loss
        
        atr_value = self.atr[0]
        current_price = self.data.close[0]
        
        # ATR-based stop (2.5 * ATR)
        atr_stop = (atr_value * 2.5) / current_price
        
        # Используем большее из фиксированного стопа или ATR стопа
        adaptive_stop = max(self.p.stop_loss, min(atr_stop, 0.05))  # Максимум 5%
        
        return adaptive_stop

    def next(self):
        # Защита от недостаточного количества данных
        if len(self.data) < max(self.p.ema_trend, self.p.rsi_period, self.p.macd_slow, self.p.market_regime_period):
            return

        # Пропускаем если есть активный ордер
        if self.order:
            return

        current_price = self.data.close[0]
        if not current_price or current_price <= 0:
            return
            
        # КРИТИЧНО: Мониторинг рисков и обновление состояния
        self._update_drawdown_metrics()
        self.market_regime = self._detect_market_regime()
        self.portfolio_heat = self._calculate_portfolio_heat()
        
        # Экстренный выход при критической просадке
        if self.position and self.emergency_mode and self.current_drawdown >= self.p.emergency_dd_threshold:
            self.order = self.close()
            self.log(f"🚨 ЭКСТРЕННЫЙ ВЫХОД! Просадка: {self.current_drawdown*100:.1f}%")
            self.trades_since_emergency = 0
            return

        # ВХОД В ПОЗИЦИЮ - Строгий контроль рисков + Kelly Criterion
        if not self.position:
            # Блокировка входов в экстренном режиме
            if self.emergency_mode and self.trades_since_emergency < 5:
                return
            
            # Проверка качества сигнала
            signal_quality = self._calculate_signal_quality()
            
            if signal_quality >= self.p.signal_quality_min:
                # Расчет позиции с Kelly Criterion и риск-контролем
                target_percent = self._calculate_advanced_position_size(signal_quality)
                
                # Проверка портфельной нагретости
                if self.portfolio_heat > self.p.max_portfolio_heat:
                    self.log(f"⚠️ Портфель перегрет ({self.portfolio_heat*100:.1f}%) - пропуск сигнала")
                    return
                
                if target_percent > 0.02:  # Минимум 2% для входа
                    # Расчет адаптивного стоп-лосса
                    adaptive_stop = self._calculate_adaptive_stops()
                    
                    self.order = self.order_target_percent(target=target_percent)
                    self.entry_price = current_price
                    self.highest_price = current_price
                    self.trailing_stop_price = None
                    self.last_entry_time = len(self.data)  # Track entry time
                    
                    # Обновляем статистику
                    if self.emergency_mode:
                        self.trades_since_emergency += 1
                        if self.trades_since_emergency >= 5:
                            self.emergency_mode = False  # Выход из экстренного режима
                    
                    portfolio_value = self.broker.getvalue()
                    kelly_info = ""
                    if self.p.use_kelly_criterion and len(self.trade_history) >= 10:
                        kelly_info = f"Kelly: {self.kelly_fraction*100:.1f}%, WR: {self.current_win_rate*100:.1f}%"
                    
                    self.log(f"📈 ПОКУПКА: {current_price:.2f}, Качество: {signal_quality:.1f}/10, RSI: {self.rsi[0]:.1f}")
                    self.log(f"🎯 Позиция: {target_percent*100:.1f}%, Режим: {self.market_regime}, "
                            f"Стоп: {adaptive_stop*100:.1f}%, {kelly_info}")

        # УПРАВЛЕНИЕ ОТКРЫТОЙ ПОЗИЦИЕЙ с продвинутым риск-менеджментом
        elif self.position and self.entry_price:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Обновляем максимальную цену для трейлинга
            if current_price > self.highest_price:
                self.highest_price = current_price
                
            # Активируем трейлинг стоп при достижении цели
            if current_profit_pct >= self.p.trailing_stop:
                trailing_price = self.highest_price * (1 - self.p.trailing_dist)
                if not self.trailing_stop_price or trailing_price > self.trailing_stop_price:
                    self.trailing_stop_price = trailing_price

            # Расчет адаптивного стоп-лосса
            adaptive_stop = self._calculate_adaptive_stops()
            
            # Условия выхода с улучшенной логикой
            exit_reason = self._should_exit_advanced(current_price, current_profit_pct, adaptive_stop)
            
            if exit_reason:
                self.order = self.close()
                self.log(f"📉 ПРОДАЖА: {current_price:.2f}, Прибыль: {current_profit_pct*100:.1f}%, Причина: {exit_reason}")
                
                # Сброс состояния
                self.entry_price = None
                self.highest_price = None
                self.trailing_stop_price = None
                self.last_entry_time = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

    def notify_trade(self, trade):
        """Обработка событий сделок с продвинутым трекингом для Kelly Criterion"""
        if not trade.isclosed:
            return
        
        pnl_pct = (trade.pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
        self.total_trades += 1
        
        # Добавляем сделку в историю для Kelly Criterion (% прибыль/убыток)
        self.trade_history.append(pnl_pct)
        
        # Ограничиваем историю для производительности
        if len(self.trade_history) > self.p.kelly_lookback * 2:
            self.trade_history = self.trade_history[-self.p.kelly_lookback:]
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0  # Сброс счетчика убытков
            self.log(f'TRADE WIN! 🎉 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')
            
            # Выход из экстренного режима после успешной сделки
            if self.emergency_mode and pnl_pct > 2.0:  # Хорошая прибыльная сделка
                self.trades_since_emergency += 2  # Ускоренный выход
        else:
            self.consecutive_losses += 1
            self.log(f'TRADE LOSS 😞 PnL: ${trade.pnl:.2f} ({pnl_pct:.2f}%) - Убытков подряд: {self.consecutive_losses}')
        
        # Пересчитываем Kelly после каждой сделки
        if len(self.trade_history) >= 10:
            self._calculate_kelly_fraction()

    def _calculate_signal_quality(self) -> float:
        """Расчет качества сигнала входа (0-10) - СТРОГАЯ ВЕРСИЯ"""
        try:
            score = 0.0
            current_price = self.data.close[0]
            
            # 1. Трендовый анализ (0-4 балла) - более строгий
            if (len(self.ema_fast) > 0 and len(self.ema_slow) > 0 and len(self.ema_trend) > 0):
                if self.ema_fast[0] > self.ema_slow[0] > self.ema_trend[0]:
                    # Проверяем силу тренда
                    trend_strength = (self.ema_fast[0] - self.ema_trend[0]) / self.ema_trend[0]
                    if trend_strength > 0.02:  # Минимум 2% разница
                        score += 3.0
                        if current_price > self.ema_trend[0] * 1.01:  # 1% выше тренда
                            score += 1.0
            
            # 2. RSI анализ с жесткими условиями (0-3 балла)
            if len(self.rsi) > 0:
                rsi_val = self.rsi[0]
                if rsi_val < self.p.rsi_oversold_strong:
                    score += 3.0  # Очень сильная перепроданность
                elif rsi_val < self.p.rsi_oversold:
                    score += 2.0  # Перепроданность
                elif 45 <= rsi_val <= 55:
                    score += 0.5  # Узкая нейтральная зона
            
            # 3. MACD анализ с дополнительными проверками (0-2 балла)
            if len(self.macd_cross_up) > 0 and self.macd_cross_up[0]:
                if len(self.macd.macd) > 0 and self.macd.macd[0] > 0:  # MACD должен быть положительным
                    score += 2.0
            elif (len(self.macd.macd) > 0 and len(self.macd.signal) > 0 and 
                  self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[0] > 0):
                score += 1.0
                
            # 4. EMA кроссы с подтверждением (0-1 балл)
            if len(self.ema_cross_up) > 0 and self.ema_cross_up[0]:
                # Дополнительное подтверждение объемом
                if (self.volume_sma and len(self.volume_sma) > 0 and 
                    hasattr(self.data, 'volume') and self.data.volume[0] > self.volume_sma[0]):
                    score += 1.0
                else:
                    score += 0.5
            
            # ЖЕСТКИЕ ШТРАФЫ за неблагоприятные условия
            if len(self.rsi) > 0 and self.rsi[0] > self.p.rsi_overbought:
                score *= 0.1  # Очень сильный штраф за перекупленность
            
            # Проверка волатильности - штраф за высокую волатильность
            if len(self.volatility) > 0 and self.volatility[0] > self.p.vol_target * 2:
                score *= 0.5
            
            # Проверка текущей просадки
            if self.current_drawdown > self.p.max_dd_threshold * 0.5:
                score *= 0.3  # Штраф во время просадки
                
            return min(max(score, 0.0), 10.0)
            
        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0

    def _update_drawdown_metrics(self):
        """Обновление метрик просадки для адаптивного управления рисками"""
        try:
            current_value = self.broker.getvalue()
            
            # Обновляем пиковое значение
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.current_drawdown = 0.0
            else:
                # Рассчитываем текущую просадку
                self.current_drawdown = (self.peak_value - current_value) / self.peak_value
                
            # Обновляем максимальную просадку
            if self.current_drawdown > self.max_drawdown_seen:
                self.max_drawdown_seen = self.current_drawdown
                
        except (ZeroDivisionError, TypeError):
            pass

    def _calculate_advanced_position_size(self, signal_quality: float) -> float:
        """Продвинутый расчет размера позиции с Kelly Criterion и строгим риск-контролем"""
        try:
            # Базовый размер позиции
            base_percent = self.p.base_position_percent
            
            # 1. Kelly Criterion размер (если достаточно истории)
            kelly_size = 0.0
            if self.p.use_kelly_criterion and len(self.trade_history) >= 10:
                kelly_size = self._calculate_kelly_fraction()
            else:
                kelly_size = base_percent * 0.5  # Консервативный подход без истории
            
            # 2. Корректировка на основе просадки (КРИТИЧНО для контроля DD)
            dd_adjustment = self._should_reduce_position_for_risk()
            
            # 3. Корректировка на качество сигнала
            if signal_quality >= 7.0:
                quality_adj = 1.3  # Исключительные сигналы
            elif signal_quality >= 5.0:
                quality_adj = 1.1  # Хорошие сигналы
            elif signal_quality >= 4.0:
                quality_adj = 1.0  # Стандартные сигналы
            else:
                quality_adj = 0.7  # Слабые сигналы
            
            # 4. Волатильность-базированная корректировка
            vol_adj = 1.0
            if len(self.volatility) > 0 and self.volatility[0] > 0:
                vol_adj = min(1.5, self.p.vol_target / max(self.volatility[0], 0.001))
                vol_adj = max(0.3, vol_adj)  # Ограничения
            
            # 5. Режим рынка корректировка
            regime_adj = 1.0
            if self.market_regime == "ranging":
                regime_adj = self.p.ranging_reduction
            elif self.market_regime in ["trending_up", "trending_down"]:
                regime_adj = 1.1  # Немного больше в трендах
            
            # Комбинируем Kelly с остальными факторами
            if kelly_size > 0:
                # Используем Kelly как основу, корректируя другими факторами
                target_percent = kelly_size * quality_adj * vol_adj * regime_adj * dd_adjustment
            else:
                # Используем традиционный подход
                target_percent = base_percent * quality_adj * vol_adj * regime_adj * dd_adjustment
            
            # Жесткие ограничения безопасности
            target_percent = min(target_percent, self.p.max_position_percent)
            target_percent = max(target_percent, 0.0)
            
            # Дополнительная проверка риска на сделку
            expected_risk = target_percent * self.p.stop_loss
            if expected_risk > self.p.max_risk_per_trade:
                target_percent = self.p.max_risk_per_trade / self.p.stop_loss
            
            return target_percent
            
        except (ZeroDivisionError, TypeError, AttributeError):
            return self.p.base_position_percent * 0.3  # Очень консервативный fallback

    def _should_exit_advanced(self, current_price: float, profit_pct: float, adaptive_stop: float) -> str:
        """Продвинутая логика выхода с ЖЕСТКИМИ условиями для контроля рисков"""
        try:
            # 1. ЖЕСТКИЙ стоп-лосс - НЕ ПОДЛЕЖИТ ОБСУЖДЕНИЮ
            if profit_pct <= -adaptive_stop:
                return "STOP_LOSS"
                
            # 2. Трейлинг стоп - защищаем прибыль 
            if self.trailing_stop_price and current_price <= self.trailing_stop_price:
                return "TRAILING_STOP"
                
            # 3. ЭКСТРЕННЫЙ выход при любой значительной просадке портфеля
            if self.current_drawdown >= self.p.max_dd_threshold * 0.8:  # При 80% от лимита
                return "EMERGENCY_EXIT"
                
            # 4. Быстрый тейк-профит - фиксируем прибыль
            if profit_pct >= self.p.take_profit:
                return "TAKE_PROFIT"
            
            # 5. РАННИЙ выход при первых признаках разворота
            if len(self.rsi) > 0 and self.rsi[0] > self.p.rsi_overbought:
                if profit_pct > 0.02:  # Снижен порог до 2%
                    return "RSI_OVERBOUGHT"
                    
            if len(self.ema_cross_down) > 0 and self.ema_cross_down[0]:
                if profit_pct > 0.01:  # Снижен порог до 1%
                    return "EMA_CROSS_DOWN"
                    
            if len(self.macd_cross_down) > 0 and self.macd_cross_down[0]:
                if profit_pct > 0.015:  # Снижен порог до 1.5%
                    return "MACD_CROSS_DOWN"
            
            # 6. Выход в боковике при ЛЮБОЙ прибыли
            if self.market_regime == "ranging" and profit_pct > 0.01:
                return "RANGING_PROFIT_TAKE"
            
            # 7. Немедленный выход при высокой волатильности
            if (len(self.volatility) > 0 and self.volatility[0] > self.p.vol_target * 2 and 
                profit_pct > 0.005):  # Даже при 0.5% прибыли
                return "HIGH_VOLATILITY_EXIT"
            
            # 8. ПРИНУДИТЕЛЬНЫЙ выход по времени (не ждем долго)
            if hasattr(self, 'last_entry_time') and self.last_entry_time:
                bars_in_position = len(self.data) - self.last_entry_time
                if bars_in_position > 100 and -0.01 < profit_pct < 0.01:  # 100 баров без результата
                    return "TIME_BASED_EXIT"
                # Также выходим при длительном убытке
                elif bars_in_position > 50 and profit_pct < -0.02:
                    return "TIME_LOSS_EXIT"
            
            # 9. Bollinger Bands - быстрый выход при экстремумах
            if (len(self.bb.top) > 0 and len(self.bb.bot) > 0 and 
                current_price > self.bb.top[0] and profit_pct > 0.015):
                return "BOLLINGER_EXTREME"
            
            # 10. Защита от последовательных убытков
            if self.consecutive_losses >= 2 and profit_pct < -0.01:
                return "CONSECUTIVE_LOSS_PROTECTION"
                
            return None
            
        except (IndexError, TypeError, ZeroDivisionError, AttributeError):
            return None

    def stop(self):
        """Финальная продвинутая статистика с риск-метриками"""
        final_value = self.broker.getvalue()
        starting_cash = self.broker.startingcash
        total_return = (final_value / starting_cash - 1) * 100
        
        self.log(f'🏁 STAS Advanced Strategy Complete!')
        self.log(f'📊 Final Value: ${final_value:.2f}')
        self.log(f'📈 Total Return: {total_return:+.2f}%')
        self.log(f'📉 Maximum Drawdown: {self.max_drawdown_seen*100:.2f}%')
        
        # Расширенная статистика
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        self.log(f'🎯 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})')
        
        if len(self.trade_history) > 0:
            avg_return = np.mean(self.trade_history)
            std_return = np.std(self.trade_history)
            sharpe_approx = avg_return / max(std_return, 0.001) if std_return > 0 else 0
            self.log(f'📊 Avg Trade Return: {avg_return:.2f}%')
            self.log(f'📊 Trade Volatility: {std_return:.2f}%')
            self.log(f'📊 Approx Sharpe: {sharpe_approx:.2f}')
        
        # Kelly Criterion статистика
        if self.p.use_kelly_criterion and len(self.trade_history) >= 10:
            self.log(f'🎲 Final Kelly Fraction: {self.kelly_fraction*100:.1f}%')
            self.log(f'🎲 Win/Loss Ratio: {self.current_avg_win_loss_ratio:.2f}')
        
        # Режим и состояние
        self.log(f'🏮 Emergency Mode Triggered: {"Yes" if self.emergency_mode else "No"}')
        self.log(f'🎯 Market Regime: {self.market_regime}')
        
        # Оценка результата с учетом риска
        risk_adjusted_score = total_return / max(self.max_drawdown_seen * 100, 1)
        self.log(f'⚡ Risk-Adjusted Score: {risk_adjusted_score:.2f}')
        
        if total_return >= 500 and self.max_drawdown_seen <= 0.25:
            self.log(f'🏆 EXCELLENT! Target achieved with controlled risk!')
        elif total_return >= 200 and self.max_drawdown_seen <= 0.35:
            self.log(f'🥇 GREAT! Good returns with acceptable risk!')
        elif total_return >= 100:
            self.log(f'✅ GOOD! Positive returns achieved!')
        elif total_return > 0:
            self.log(f'📈 POSITIVE! Room for optimization!')
        else:
            self.log(f'❌ NEGATIVE! Strategy needs major revision!')