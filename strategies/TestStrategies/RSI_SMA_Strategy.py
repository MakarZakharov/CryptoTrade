import backtrader as bt
import backtrader.indicators as btind


class RSI_SMA_Strategy(bt.Strategy):
    """
    Улучшенная торговая стратегия для высокой доходности

    Стратегии:
    1. 'simple_rsi' - Простая RSI стратегия
    2. 'aggressive_momentum' - Агрессивная momentum стратегия
    3. 'trend_breakout' - Стратегия прорыва трендов
    4. 'buy_and_hold' - Стратегия покупай и держи
    5. 'mean_reversion' - Стратегия возврата к среднему
    6. 'multi_indicator' - Мульти-индикаторная стратегия
    7. 'dynamic_risk' - Стратегия с динамическими рисками
    """

    params = (
        ('rsi_period', 14),
        ('sma_period', 50),
        ('sma_fast', 20),
        ('sma_slow', 50),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_size', 0.95),
        ('printlog', False),
        ('strategy_type', 'simple_rsi'),
        # Параметры для агрессивных стратегий
        ('momentum_period', 10),
        ('volatility_period', 20),
        ('breakout_period', 20),
        ('stop_loss_pct', 0.05),  # 5% стоп-лосс
        ('take_profit_pct', 0.15),  # 15% тейк-профит
        ('trailing_stop_pct', 0.08),  # 8% трейлинг-стоп
        ('use_leverage', True),
        ('leverage_multiplier', 1.5),
    )

    def log(self, txt, dt=None, doprint=False):
        """Функция логирования"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        """Инициализация индикаторов"""
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # Основные индикаторы
        self.rsi = btind.RSI(self.datas[0], period=self.params.rsi_period)
        self.sma = btind.SimpleMovingAverage(self.datas[0], period=self.params.sma_period)
        self.sma_fast = btind.SimpleMovingAverage(self.datas[0], period=self.params.sma_fast)
        self.sma_slow = btind.SimpleMovingAverage(self.datas[0], period=self.params.sma_slow)

        # Дополнительные индикаторы
        self.ema_fast = btind.ExponentialMovingAverage(self.datas[0], period=12)
        self.ema_slow = btind.ExponentialMovingAverage(self.datas[0], period=26)
        self.macd = btind.MACD(self.datas[0])
        self.bollinger = btind.BollingerBands(self.datas[0], period=20)
        self.momentum = btind.Momentum(self.datas[0], period=self.params.momentum_period)
        self.roc = btind.RateOfChange(self.datas[0], period=10)

        # Для стратегии прорыва - НЕ создаем сигналы в __init__
        if self.params.strategy_type == 'trend_breakout':
            self.highest = btind.Highest(self.datahigh, period=self.params.breakout_period)
            self.lowest = btind.Lowest(self.datalow, period=self.params.breakout_period)
            self.volume_sma = btind.SimpleMovingAverage(self.datavolume, period=20)

        # Настройка сигналов для других стратегий
        self._setup_strategy_signals()

        # Переменные для отслеживания
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.buy_and_hold_executed = False

    def _setup_strategy_signals(self):
        """Настройка сигналов для разных стратегий"""

        if self.params.strategy_type == 'simple_rsi':
            # Простая RSI стратегия - должна давать много сделок
            self.buy_signal = self.rsi < self.params.rsi_oversold
            self.sell_signal = self.rsi > self.params.rsi_overbought

        elif self.params.strategy_type == 'aggressive_momentum':
            # Агрессивная momentum стратегия - более мягкие условия
            self.buy_signal = bt.Or(
                bt.And(self.rsi < 45, self.momentum > 0),
                bt.And(self.dataclose > self.sma_fast, self.rsi < 50)
            )
            self.sell_signal = bt.Or(
                self.rsi > 70,
                bt.And(self.dataclose < self.sma_fast, self.rsi > 55)
            )

        elif self.params.strategy_type == 'multi_indicator':
            # Мульти-индикаторная стратегия - используем ИЛИ вместо И для входа
            self.buy_signal = bt.Or(
                self.rsi < self.params.rsi_oversold,  # RSI перепродан
                bt.And(self.dataclose > self.sma_fast, self.sma_fast > self.sma_slow),  # Восходящий тренд
                self.dataclose < self.bollinger.lines.bot  # Цена ниже нижней полосы Боллинджера
            )
            self.sell_signal = bt.Or(
                self.rsi > self.params.rsi_overbought,
                self.dataclose < self.sma_fast,
                self.dataclose > self.bollinger.lines.top
            )

        elif self.params.strategy_type == 'dynamic_risk':
            # Стратегия с динамическими рисками - простые условия
            self.buy_signal = bt.Or(
                self.rsi < 40,  # RSI ниже 40
                bt.And(self.momentum > 0, self.dataclose > self.sma),  # Положительный моментум + цена выше SMA
                self.roc > 5  # Сильный положительный ROC
            )
            self.sell_signal = bt.Or(
                self.rsi > 65,
                bt.And(self.momentum < 0, self.dataclose < self.sma),
                self.roc < -5
            )

        elif self.params.strategy_type == 'mean_reversion':
            # Стратегия возврата к среднему
            self.buy_signal = bt.Or(
                self.rsi < 35,
                self.dataclose < self.bollinger.lines.bot
            )

            self.sell_signal = bt.Or(
                self.rsi > 65,
                self.dataclose > self.bollinger.lines.top
            )

        elif self.params.strategy_type == 'buy_and_hold':
            # Простая стратегия покупай и держи
            pass  # Логика в next()

        # trend_breakout обрабатывается в next() из-за индикаторов

        elif self.params.strategy_type == 'multi_indicator':
            # Мульти-индикаторная стратегия
            self.buy_signal = bt.And(
                self.rsi < 30,
                self.dataclose < self.sma_slow,
                self.momentum > 0
            )
            self.sell_signal = bt.Or(
                self.rsi > 70,
                self.dataclose > self.sma_slow,
                self.momentum < 0
            )

        elif self.params.strategy_type == 'dynamic_risk':
            # Стратегия с динамическими рисками
            self.buy_signal = bt.And(
                self.rsi < 30,
                self.dataclose < self.sma_slow,
                self.momentum > 0
            )
            self.sell_signal = bt.Or(
                self.rsi > 70,
                self.dataclose > self.sma_slow,
                self.momentum < 0
            )

    def notify_order(self, order):
        """Уведомления об ордерах"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'ПОКУПКА ВЫПОЛНЕНА, Цена: {order.executed.price:.2f}, '
                    f'Стоимость: {order.executed.value:.2f}, '
                    f'Комиссия: {order.executed.comm:.2f}', doprint=True
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

                # Устанавливаем стоп-лосс и тейк-профит
                self.stop_loss_price = self.buyprice * (1 - self.params.stop_loss_pct)
                self.take_profit_price = self.buyprice * (1 + self.params.take_profit_pct)
                if hasattr(self.params, 'trailing_stop_pct'):
                    self.trailing_stop_price = self.buyprice * (1 - self.params.trailing_stop_pct)

            else:
                self.log(
                    f'ПРОДАЖА ВЫПОЛНЕНА, Цена: {order.executed.price:.2f}, '
                    f'Стоимость: {order.executed.value:.2f}, '
                    f'Комиссия: {order.executed.comm:.2f}', doprint=True
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Ордер отменен/отклонен/нет маржи', doprint=True)

        self.order = None

    def notify_trade(self, trade):
        """Уведомления о сделках"""
        if not trade.isclosed:
            return

        self.log(f'ОПЕРАЦИЯ ПРИБЫЛЬ, Общая: {trade.pnl:.2f}, Чистая: {trade.pnlcomm:.2f}', doprint=True)

        # Отслеживаем последовательные победы/поражения
        if trade.pnlcomm > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def _calculate_position_size(self):
        """Расчет размера позиции"""
        base_position_size = self.params.position_size

        # Увеличиваем размер позиции после побед
        if self.consecutive_wins >= 2:
            base_position_size = min(0.99, base_position_size * 1.1)

        # Применяем имитацию левереджа
        if self.params.use_leverage:
            base_position_size *= self.params.leverage_multiplier

        return min(0.99, base_position_size)

    def _check_exit_conditions(self):
        """Проверка условий выхода из позиции"""
        if not self.buyprice:
            return False

        current_price = self.dataclose[0]

        # Стоп-лосс
        if self.stop_loss_price and current_price <= self.stop_loss_price:
            self.log(f'СТОП-ЛОСС: {current_price:.2f} <= {self.stop_loss_price:.2f}', doprint=True)
            return True

        # Тейк-профит
        if self.take_profit_price and current_price >= self.take_profit_price:
            self.log(f'ТЕЙК-ПРОФИТ: {current_price:.2f} >= {self.take_profit_price:.2f}', doprint=True)
            return True

        return False

    def next(self):
        """Основная логика стратегии"""
        # Включаем подробное логирование только для отладки
        debug_logging = self.params.printlog

        if debug_logging:
            self.log(f'Закрытие: {self.dataclose[0]:.2f}, RSI: {self.rsi[0]:.2f}, '
                    f'SMA: {self.sma[0]:.2f}')

        # Если есть активный ордер, ждем его исполнения
        if self.order:
            return

        # Если есть позиция, проверяем условия выхода
        if self.position:
            # Проверяем стоп-лосс, тейк-профит
            if self._check_exit_conditions():
                self.order = self.sell(size=self.position.size)
                return

            # Проверяем сигнал на продажу для разных стратегий
            should_sell = False

            if self.params.strategy_type == 'buy_and_hold':
                # Никогда не продаем для buy_and_hold
                should_sell = False
            elif self.params.strategy_type == 'trend_breakout':
                # Динамические сигналы для прорыва
                if (len(self.lowest) > 0 and self.dataclose[0] < self.lowest[0]) or self.rsi[0] < 30:
                    should_sell = True
            elif hasattr(self, 'sell_signal'):
                should_sell = self.sell_signal[0]

            if should_sell:
                self.log(f'СИГНАЛ ПРОДАЖИ: RSI={self.rsi[0]:.2f}, Стратегия={self.params.strategy_type}', doprint=True)
                self.order = self.sell(size=self.position.size)

        else:
            # Если нет позиции, ищем сигнал на покупку
            should_buy = False

            if self.params.strategy_type == 'buy_and_hold':
                # Покупаем один раз в начале
                if not self.buy_and_hold_executed:
                    should_buy = True
                    self.buy_and_hold_executed = True
            elif self.params.strategy_type == 'trend_breakout':
                # Динамические сигналы для прорыва
                if (len(self.highest) > 0 and len(self.volume_sma) > 0 and
                    self.dataclose[0] > self.highest[0] and
                    self.rsi[0] > 50 and
                    self.datavolume[0] > self.volume_sma[0]):
                    should_buy = True
            elif hasattr(self, 'buy_signal'):
                should_buy = self.buy_signal[0]

                # Отладочная информация
                if debug_logging:
                    self.log(f'Проверка сигнала покупки: {should_buy}, '
                            f'RSI: {self.rsi[0]:.2f}, SMA_fast: {self.sma_fast[0]:.2f}, '
                            f'SMA_slow: {self.sma_slow[0]:.2f}, Momentum: {self.momentum[0]:.2f}')

            if should_buy:
                self.log(f'СИГНАЛ ПОКУПКИ: RSI={self.rsi[0]:.2f}, Стратегия={self.params.strategy_type}', doprint=True)

                # Рассчитываем размер позиции
                position_size = self._calculate_position_size()
                size = int(self.broker.getcash() * position_size / self.dataclose[0])

                if size > 0:
                    self.order = self.buy(size=size)
                else:
                    self.log(f'Недостаточно средств для покупки. Размер: {size}', doprint=True)

    def stop(self):
        """Вызывается в конце бэктеста"""
        final_value = self.broker.getvalue()
        total_return = (final_value - 10000) / 10000 * 100
        self.log(f'Итоговая стоимость портфеля: {final_value:.2f}', doprint=True)
        self.log(f'Общая доходность: {total_return:.2f}%', doprint=True)
        self.log(f'Стратегия: {self.params.strategy_type}', doprint=True)