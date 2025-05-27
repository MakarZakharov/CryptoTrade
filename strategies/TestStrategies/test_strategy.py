import backtrader as bt


class ImprovedHFT_Strategy(bt.Strategy):
    """Покращена високочастотна стратегія з оптимізованим кодом"""

    params = (
        # Швидкі індикатори
        ("ema_fast", 3), ("ema_slow", 8), ("rsi_period", 5),
        ("macd_fast", 3), ("macd_slow", 8), ("macd_signal", 4),

        # Позиційні параметри
        ("position_size", 0.8), ("stop_loss", 0.008), ("take_profit", 0.015),
        ("max_hold_bars", 3), ("signal_threshold", 2),

        # Фільтри ризику
        ("min_volume_ratio", 0.5), ("max_spread_pct", 0.002),
        ("min_price_move", 0.0005), ("cooldown_bars", 1)
    )

    def __init__(self):
        # Основні індикатори
        self.ema_fast = bt.indicators.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(period=self.p.ema_slow)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

        # Сигнали кросоверів
        self.ema_cross_up = bt.indicators.CrossUp(self.ema_fast, self.ema_slow)
        self.ema_cross_down = bt.indicators.CrossDown(self.ema_fast, self.ema_slow)
        self.macd_cross_up = bt.indicators.CrossUp(self.macd.macd, self.macd.signal)
        self.macd_cross_down = bt.indicators.CrossDown(self.macd.macd, self.macd.signal)

        # Додаткові швидкі індикатори
        self.momentum = bt.indicators.Momentum(period=2)
        self.atr = bt.indicators.ATR(period=5)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=10)

        # Стан стратегії
        self.reset_position_state()
        self.trade_stats = {'trades': 0, 'wins': 0, 'signals': 0, 'last_trade_bar': -999}

    def reset_position_state(self):
        self.entry_price = self.entry_bar = self.position_type = None

    def next(self):
        if len(self.data) < 15:  # Мінімальна кількість даних
            return

        if self.position:
            self._manage_position()
        else:
            self._scan_entry_signals()

    def _scan_entry_signals(self):
        """Швидке сканування сигналів входу"""
        current_bar = len(self.data)

        # Кулдаун між угодами
        if current_bar <= self.trade_stats['last_trade_bar'] + self.p.cooldown_bars:
            return

        # Швидкі фільтри ліквідності
        if not self._liquidity_filter():
            return

        signals = self._calculate_signals()
        self.trade_stats['signals'] += 1

        # Вхід по сигналах
        if signals['buy'] >= self.p.signal_threshold and signals['buy'] > signals['sell']:
            self._enter_long(signals['buy'])
        elif signals['sell'] >= self.p.signal_threshold and signals['sell'] > signals['buy']:
            self._enter_short(signals['sell'])

    def _liquidity_filter(self):
        """Швидка перевірка ліквідності"""
        # Обсяг
        if len(self.volume_sma) > 0 and self.data.volume[0] < self.volume_sma[0] * self.p.min_volume_ratio:
            return False

        # Спред (високий-низький)
        spread_pct = (self.data.high[0] - self.data.low[0]) / self.data.close[0]
        if spread_pct > self.p.max_spread_pct:
            return False

        # Мінімальний рух ціни
        if len(self.data.close) > 1:
            price_move = abs(self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            if price_move < self.p.min_price_move:
                return False

        return True

    def _calculate_signals(self):
        """Компактний розрахунок сигналів"""
        buy = sell = 0

        # Кросовери (найважливіші)
        if self.ema_cross_up[0]: buy += 3
        if self.ema_cross_down[0]: sell += 3
        if self.macd_cross_up[0]: buy += 2
        if self.macd_cross_down[0]: sell += 2

        # Поточні тренди
        if self.ema_fast[0] > self.ema_slow[0]:
            buy += 1
        else:
            sell += 1

        if self.macd.macd[0] > self.macd.signal[0]:
            buy += 1
        else:
            sell += 1

        # RSI фільтр (не екстремальні значення)
        if 25 < self.rsi[0] < 75:
            if self.rsi[0] < 45:
                buy += 1
            elif self.rsi[0] > 55:
                sell += 1

        # Momentum
        if self.momentum[0] > 0:
            buy += 1
        else:
            sell += 1

        return {'buy': buy, 'sell': sell}

    def _enter_long(self, signal_strength):
        size = self._calculate_position_size()
        if size > 0:
            self.buy(size=size)
            self._set_position_state('LONG', len(self.data))
            print(f"🟢 LONG #{self.trade_stats['trades']}: ${self.data.close[0]:.4f} | "
                  f"Size: {size} | Signals: {signal_strength}")

    def _enter_short(self, signal_strength):
        size = self._calculate_position_size()
        if size > 0:
            self.sell(size=size)
            self._set_position_state('SHORT', len(self.data))
            print(f"🔴 SHORT #{self.trade_stats['trades']}: ${self.data.close[0]:.4f} | "
                  f"Size: {size} | Signals: {signal_strength}")

    def _calculate_position_size(self):
        """Динамічний розрахунок розміру позиції з урахуванням волатильності"""
        cash = self.broker.get_cash()
        price = self.data.close[0]

        # Базовий розмір
        base_size = int((cash * self.p.position_size) / price)

        # Корекція на волатільність (ATR)
        if len(self.atr) > 0:
            volatility_factor = min(1.0, 0.02 / (self.atr[0] / price))  # Зменшуємо розмір при високій волатільності
            adjusted_size = int(base_size * volatility_factor)
            return max(adjusted_size, 1)

        return base_size

    def _set_position_state(self, pos_type, bar):
        self.entry_price = self.data.close[0]
        self.entry_bar = bar
        self.position_type = pos_type
        self.trade_stats['trades'] += 1
        self.trade_stats['last_trade_bar'] = bar

    def _manage_position(self):
        """Ефективне управління позицією"""
        if not self.entry_price:
            return

        price = self.data.close[0]
        bars_held = len(self.data) - self.entry_bar

        # Розрахунок прибутку
        if self.position_type == 'LONG':
            profit_pct = (price - self.entry_price) / self.entry_price
            exit_signals = [
                (profit_pct <= -self.p.stop_loss, "Stop Loss"),
                (profit_pct >= self.p.take_profit, "Take Profit"),
                (bars_held >= self.p.max_hold_bars, "Max Hold"),
                (self.ema_cross_down[0] or self.macd_cross_down[0], "Reversal")
            ]
        else:  # SHORT
            profit_pct = (self.entry_price - price) / self.entry_price
            exit_signals = [
                (profit_pct <= -self.p.stop_loss, "Stop Loss"),
                (profit_pct >= self.p.take_profit, "Take Profit"),
                (bars_held >= self.p.max_hold_bars, "Max Hold"),
                (self.ema_cross_up[0] or self.macd_cross_up[0], "Reversal")
            ]

        # Перевірка умов виходу
        for condition, reason in exit_signals:
            if condition:
                self._exit_position(price, profit_pct, reason, bars_held)
                break

    def _exit_position(self, price, profit_pct, reason, bars_held):
        self.close()

        if profit_pct > 0:
            self.trade_stats['wins'] += 1

        emoji = "📈" if profit_pct > 0 else "📉"
        print(f"⚡ CLOSE {self.position_type}: ${price:.4f} | {emoji} {profit_pct * 100:+.2f}% | "
              f"{reason} | {bars_held}bars")

        self.reset_position_state()

    def stop(self):
        """Компактна статистика"""
        final_value = self.broker.get_value()
        initial_cash = self.broker.startingcash  # Використовуємо початковий капітал брокера
        total_return = (final_value - initial_cash) / initial_cash * 100
        win_rate = (self.trade_stats['wins'] / self.trade_stats['trades'] * 100) if self.trade_stats[
                                                                                        'trades'] > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"🚀 ПОКРАЩЕНА HFT СТРАТЕГІЯ - РЕЗУЛЬТАТИ")
        print(f"{'=' * 50}")
        print(f"💰 Початковий капітал: ${initial_cash:,.2f}")
        print(f"💰 Фінальний капітал: ${final_value:,.2f}")
        print(f"💰 P&L: ${final_value - initial_cash:+,.2f} ({total_return:+.2f}%)")
        print(f"🔄 Угод: {self.trade_stats['trades']} | Win Rate: {win_rate:.1f}%")
        print(f"📡 Сигналів: {self.trade_stats['signals']}")
        if self.trade_stats['trades'] > 0:
            print(f"💹 Середнє на угоду: {total_return / self.trade_stats['trades']:.3f}%")
        print(f"{'=' * 50}")

        # Повертаємо результати для подальшого використання
        return {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'profit_loss': final_value - initial_cash,
            'total_return_pct': total_return,
            'total_trades': self.trade_stats['trades'],
            'wins': self.trade_stats['wins'],
            'win_rate_pct': win_rate,
            'total_signals': self.trade_stats['signals'],
            'avg_profit_per_trade_pct': total_return / self.trade_stats['trades'] if self.trade_stats[
                                                                                         'trades'] > 0 else 0
        }

    def get_performance_metrics(self):
        """Отримати поточні метрики продуктивності"""
        current_value = self.broker.get_value()
        initial_cash = self.broker.startingcash
        current_return = (current_value - initial_cash) / initial_cash * 100

        return {
            'current_value': current_value,
            'current_return_pct': current_return,
            'current_profit_loss': current_value - initial_cash,
            'trades_count': self.trade_stats['trades'],
            'win_rate_pct': (self.trade_stats['wins'] / self.trade_stats['trades'] * 100) if self.trade_stats[
                                                                                                 'trades'] > 0 else 0
        }