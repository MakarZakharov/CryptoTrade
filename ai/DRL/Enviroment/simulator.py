"""
Модуль симуляции рынка с реалистичной моделью исполнения ордеров.
Включает проскальзывание, spread, частичное исполнение, комиссии.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time


class OrderSide(Enum):
    """Сторона ордера."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Тип ордера."""
    MARKET = "market"
    LIMIT = "limit"


class SlippageModel(Enum):
    """Модель проскальзывания."""
    FIXED = "fixed"  # Фиксированное проскальзывание
    PERCENTAGE = "percentage"  # Процентное от цены
    VOLUME_BASED = "volume_based"  # Зависит от объема
    ELLIPTIC = "elliptic"  # Эллиптическая модель (увеличивается нелинейно)


@dataclass
class OrderResult:
    """Результат исполнения ордера."""
    executed: bool
    side: OrderSide
    quantity: float
    executed_quantity: float
    price: float
    executed_price: float
    commission: float
    slippage: float
    timestamp: float
    partial_fill: bool
    fill_ratio: float
    total_cost: float  # Общая стоимость включая комиссию


@dataclass
class MarketState:
    """Текущее состояние рынка."""
    timestamp: float
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    spread_pct: float
    volume: float
    liquidity_factor: float


class MarketSimulator:
    """
    Симулятор рыночного исполнения ордеров.

    Включает:
    - Проскальзывание (slippage)
    - Спред (bid-ask spread)
    - Частичное исполнение
    - Комиссии (maker/taker)
    - Влияние ликвидности
    - Задержки исполнения
    """

    def __init__(
        self,
        # Комиссии
        maker_fee: float = 0.0001,  # 0.01% maker
        taker_fee: float = 0.001,   # 0.1% taker

        # Спред
        spread_base: float = 0.0001,  # Базовый спред 0.01%
        spread_volatility_multiplier: float = 1.0,

        # Проскальзывание
        slippage_model: SlippageModel = SlippageModel.PERCENTAGE,
        slippage_fixed: float = 0.0,  # Фиксированное в USD
        slippage_percentage: float = 0.0005,  # 0.05%
        slippage_impact_factor: float = 0.01,  # Влияние объема

        # Ликвидность
        liquidity_factor: float = 1.0,  # 1.0 = нормальная ликвидность
        min_liquidity_factor: float = 0.1,

        # Лимиты
        min_order_size: float = 0.001,  # Минимальный размер ордера
        max_order_size: Optional[float] = None,

        # Частичное исполнение
        allow_partial_fills: bool = True,
        partial_fill_threshold: float = 0.8,  # Заполнить минимум 80%

        # Задержки
        execution_delay_ms: float = 0.0,  # Задержка исполнения в мс

        # Другое
        tick_size: float = 0.01,  # Минимальный шаг цены
        random_seed: Optional[int] = None
    ):
        """
        Инициализация симулятора рынка.

        Args:
            maker_fee: Комиссия мейкера (долевая)
            taker_fee: Комиссия тейкера (долевая)
            spread_base: Базовый спред
            spread_volatility_multiplier: Множитель спреда в зависимости от волатильности
            slippage_model: Модель проскальзывания
            slippage_fixed: Фиксированное проскальзывание
            slippage_percentage: Процентное проскальзывание
            slippage_impact_factor: Фактор влияния объема на проскальзывание
            liquidity_factor: Фактор ликвидности рынка
            min_liquidity_factor: Минимальный фактор ликвидности
            min_order_size: Минимальный размер ордера
            max_order_size: Максимальный размер ордера
            allow_partial_fills: Разрешить частичное исполнение
            partial_fill_threshold: Порог для частичного исполнения
            execution_delay_ms: Задержка исполнения
            tick_size: Минимальный шаг цены
            random_seed: Seed для воспроизводимости
        """
        # Комиссии
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # Спред
        self.spread_base = spread_base
        self.spread_volatility_multiplier = spread_volatility_multiplier

        # Проскальзывание
        self.slippage_model = slippage_model
        self.slippage_fixed = slippage_fixed
        self.slippage_percentage = slippage_percentage
        self.slippage_impact_factor = slippage_impact_factor

        # Ликвидность
        self.liquidity_factor = liquidity_factor
        self.min_liquidity_factor = min_liquidity_factor

        # Лимиты
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size

        # Частичное исполнение
        self.allow_partial_fills = allow_partial_fills
        self.partial_fill_threshold = partial_fill_threshold

        # Задержки
        self.execution_delay_ms = execution_delay_ms

        # Другое
        self.tick_size = tick_size

        # Random state
        self.rng = np.random.RandomState(random_seed)

        # История ордеров
        self.order_history: List[OrderResult] = []

    def get_market_state(
        self,
        mid_price: float,
        volume: float,
        volatility: Optional[float] = None
    ) -> MarketState:
        """
        Получить текущее состояние рынка.

        Args:
            mid_price: Средняя цена
            volume: Объем торгов
            volatility: Волатильность (опционально)

        Returns:
            MarketState с текущим состоянием рынка
        """
        # Расчет спреда
        if volatility is not None:
            spread_pct = self.spread_base * (1 + volatility * self.spread_volatility_multiplier)
        else:
            spread_pct = self.spread_base

        spread = mid_price * spread_pct

        # Bid и Ask цены
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2

        # Округление до tick_size
        bid_price = self._round_to_tick(bid_price)
        ask_price = self._round_to_tick(ask_price)

        # Фактор ликвидности (упрощенная модель)
        avg_volume = max(volume, 1.0)
        liquidity = max(self.liquidity_factor * (volume / avg_volume), self.min_liquidity_factor)

        return MarketState(
            timestamp=time.time(),
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            spread=spread,
            spread_pct=spread_pct,
            volume=volume,
            liquidity_factor=liquidity
        )

    def execute_order(
        self,
        side: OrderSide,
        quantity: float,
        market_state: MarketState,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> OrderResult:
        """
        Исполнить ордер с учетом всех рыночных условий.

        Args:
            side: Сторона ордера (BUY/SELL)
            quantity: Количество
            market_state: Текущее состояние рынка
            order_type: Тип ордера
            limit_price: Лимитная цена (для лимитных ордеров)

        Returns:
            OrderResult с результатом исполнения
        """
        # Задержка исполнения
        if self.execution_delay_ms > 0:
            time.sleep(self.execution_delay_ms / 1000.0)

        # Проверка минимального размера
        if quantity < self.min_order_size:
            return OrderResult(
                executed=False,
                side=side,
                quantity=quantity,
                executed_quantity=0.0,
                price=0.0,
                executed_price=0.0,
                commission=0.0,
                slippage=0.0,
                timestamp=time.time(),
                partial_fill=False,
                fill_ratio=0.0,
                total_cost=0.0
            )

        # Проверка максимального размера
        if self.max_order_size is not None and quantity > self.max_order_size:
            quantity = self.max_order_size

        # Базовая цена для исполнения
        if side == OrderSide.BUY:
            base_price = market_state.ask_price
        else:
            base_price = market_state.bid_price

        # Проверка лимитной цены
        if order_type == OrderType.LIMIT and limit_price is not None:
            if side == OrderSide.BUY and limit_price < base_price:
                # Лимит покупки ниже ask - не исполняется
                return self._create_unfilled_order(side, quantity)
            elif side == OrderSide.SELL and limit_price > base_price:
                # Лимит продажи выше bid - не исполняется
                return self._create_unfilled_order(side, quantity)
            base_price = limit_price

        # Расчет проскальзывания
        slippage = self._calculate_slippage(
            quantity=quantity,
            base_price=base_price,
            market_state=market_state
        )

        # Цена исполнения с учетом проскальзывания
        if side == OrderSide.BUY:
            executed_price = base_price + slippage
        else:
            executed_price = base_price - slippage

        executed_price = self._round_to_tick(executed_price)

        # Частичное исполнение
        executed_quantity = quantity
        partial_fill = False

        if self.allow_partial_fills:
            # Симулируем частичное исполнение на основе ликвидности
            fill_probability = min(1.0, market_state.liquidity_factor)

            if self.rng.random() > fill_probability:
                # Частичное исполнение
                fill_ratio = self.rng.uniform(self.partial_fill_threshold, 1.0)
                executed_quantity = quantity * fill_ratio
                partial_fill = True

        # Комиссия
        if order_type == OrderType.MARKET:
            fee_rate = self.taker_fee
        else:
            fee_rate = self.maker_fee

        commission = executed_quantity * executed_price * fee_rate

        # Общая стоимость
        total_cost = executed_quantity * executed_price + commission

        # Создаем результат
        result = OrderResult(
            executed=True,
            side=side,
            quantity=quantity,
            executed_quantity=executed_quantity,
            price=base_price,
            executed_price=executed_price,
            commission=commission,
            slippage=slippage,
            timestamp=time.time(),
            partial_fill=partial_fill,
            fill_ratio=executed_quantity / quantity if quantity > 0 else 0.0,
            total_cost=total_cost
        )

        # Сохраняем в историю
        self.order_history.append(result)

        return result

    def _calculate_slippage(
        self,
        quantity: float,
        base_price: float,
        market_state: MarketState
    ) -> float:
        """
        Рассчитать проскальзывание в зависимости от модели.

        Args:
            quantity: Количество в ордере
            base_price: Базовая цена
            market_state: Состояние рынка

        Returns:
            Проскальзывание в абсолютных единицах
        """
        if self.slippage_model == SlippageModel.FIXED:
            return self.slippage_fixed

        elif self.slippage_model == SlippageModel.PERCENTAGE:
            return base_price * self.slippage_percentage

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Проскальзывание зависит от размера ордера относительно объема
            volume_ratio = quantity * base_price / max(market_state.volume, 1.0)
            impact = volume_ratio * self.slippage_impact_factor
            return base_price * impact / market_state.liquidity_factor

        elif self.slippage_model == SlippageModel.ELLIPTIC:
            # Нелинейное увеличение проскальзывания
            volume_ratio = quantity * base_price / max(market_state.volume, 1.0)
            impact = np.sqrt(1 + volume_ratio) - 1
            return base_price * impact * self.slippage_impact_factor / market_state.liquidity_factor

        else:
            return 0.0

    def _round_to_tick(self, price: float) -> float:
        """Округлить цену до tick_size."""
        return round(price / self.tick_size) * self.tick_size

    def _create_unfilled_order(self, side: OrderSide, quantity: float) -> OrderResult:
        """Создать неисполненный ордер."""
        return OrderResult(
            executed=False,
            side=side,
            quantity=quantity,
            executed_quantity=0.0,
            price=0.0,
            executed_price=0.0,
            commission=0.0,
            slippage=0.0,
            timestamp=time.time(),
            partial_fill=False,
            fill_ratio=0.0,
            total_cost=0.0
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику исполненных ордеров.

        Returns:
            Словарь со статистикой
        """
        if not self.order_history:
            return {
                'total_orders': 0,
                'executed_orders': 0,
                'partial_fills': 0,
                'avg_slippage': 0.0,
                'avg_commission': 0.0,
                'avg_fill_ratio': 0.0
            }

        executed = [o for o in self.order_history if o.executed]
        partial = [o for o in executed if o.partial_fill]

        return {
            'total_orders': len(self.order_history),
            'executed_orders': len(executed),
            'partial_fills': len(partial),
            'avg_slippage': np.mean([o.slippage for o in executed]) if executed else 0.0,
            'avg_commission': np.mean([o.commission for o in executed]) if executed else 0.0,
            'avg_fill_ratio': np.mean([o.fill_ratio for o in executed]) if executed else 0.0,
            'total_commission': sum([o.commission for o in executed]),
            'total_slippage': sum([o.slippage for o in executed])
        }

    def reset_history(self):
        """Сбросить историю ордеров."""
        self.order_history = []

    def set_seed(self, seed: int):
        """Установить seed для воспроизводимости."""
        self.rng = np.random.RandomState(seed)


if __name__ == "__main__":
    # Тестирование симулятора
    print("=== Market Simulator Test ===\n")

    # Создаем симулятор
    simulator = MarketSimulator(
        maker_fee=0.0001,
        taker_fee=0.001,
        slippage_model=SlippageModel.PERCENTAGE,
        slippage_percentage=0.0005,
        allow_partial_fills=True,
        random_seed=42
    )

    # Текущее состояние рынка
    market = simulator.get_market_state(
        mid_price=50000.0,
        volume=1000000.0,
        volatility=0.02
    )

    print(f"Market State:")
    print(f"  Mid Price: ${market.mid_price:.2f}")
    print(f"  Bid: ${market.bid_price:.2f}")
    print(f"  Ask: ${market.ask_price:.2f}")
    print(f"  Spread: ${market.spread:.2f} ({market.spread_pct*100:.4f}%)")
    print(f"  Liquidity Factor: {market.liquidity_factor:.2f}\n")

    # Исполняем несколько ордеров
    print("Executing orders:\n")

    # Buy order
    buy_result = simulator.execute_order(
        side=OrderSide.BUY,
        quantity=0.5,
        market_state=market
    )

    print(f"BUY Order:")
    print(f"  Executed: {buy_result.executed}")
    print(f"  Quantity: {buy_result.executed_quantity:.4f}")
    print(f"  Price: ${buy_result.executed_price:.2f}")
    print(f"  Commission: ${buy_result.commission:.2f}")
    print(f"  Slippage: ${buy_result.slippage:.2f}")
    print(f"  Total Cost: ${buy_result.total_cost:.2f}\n")

    # Sell order
    sell_result = simulator.execute_order(
        side=OrderSide.SELL,
        quantity=0.3,
        market_state=market
    )

    print(f"SELL Order:")
    print(f"  Executed: {sell_result.executed}")
    print(f"  Quantity: {sell_result.executed_quantity:.4f}")
    print(f"  Price: ${sell_result.executed_price:.2f}")
    print(f"  Commission: ${sell_result.commission:.2f}")
    print(f"  Slippage: ${sell_result.slippage:.2f}")
    print(f"  Total Cost: ${sell_result.total_cost:.2f}\n")

    # Статистика
    stats = simulator.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
