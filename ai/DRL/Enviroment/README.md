# 🚀 Crypto Trading DRL Environment

Полнофункциональное торговое окружение для обучения Deep Reinforcement Learning (DRL) агентов на криптовалютных данных. Совместимо с **Gymnasium** и **Stable-Baselines3**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Оглавление

- [Особенности](#-особенности)
- [Быстрый старт](#-быстрый-старт)
- [Архитектура](#-архитектура)
- [Использование](#-использование)
- [Конфигурация](#-конфигурация)
- [Dashboard](#-интерактивная-панель)
- [Обучение агента](#-обучение-агента)
- [API Reference](#-api-reference)
- [Примеры](#-примеры)
- [Тестирование](#-тестирование)

---

## ✨ Особенности

### 🎯 Окружение
- ✅ **Полная совместимость** с Gymnasium и Stable-Baselines3
- ✅ **Гибкие action spaces**: дискретный (Hold/Buy/Sell) и непрерывный [-1, 1]
- ✅ **Параметризуемые награды**: PnL, Log-return, Sharpe, Sortino, Risk-adjusted
- ✅ **Технические индикаторы**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- ✅ **Train/Val/Test splits** для честной оценки

### 📊 Реалистичная симуляция
- ✅ **Проскальзывание (slippage)**: 4 модели (fixed, percentage, volume-based, elliptic)
- ✅ **Bid-Ask spread** с динамической волатильностью
- ✅ **Комиссии**: Maker/Taker fees
- ✅ **Частичное исполнение** ордеров
- ✅ **Влияние ликвидности** на исполнение

### 📈 Метрики и визуализация
- ✅ **Комплексные метрики**: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate
- ✅ **Статические графики** (Matplotlib): Candlestick, Equity, Drawdown
- ✅ **Интерактивные графики** (Plotly): полный анализ с возможностью zoom/pan
- ✅ **Streamlit Dashboard**: real-time мониторинг и тестирование

### 🛠 Удобство использования
- ✅ **Автоматическая загрузка данных** из Parquet/CSV
- ✅ **Конфигурация через YAML/JSON**
- ✅ **Unit-тесты** для всех компонентов
- ✅ **Jupyter notebooks** с примерами обучения
- ✅ **Детальная документация**

---

## 🚀 Быстрый старт

### 1. Активация виртуального окружения

```bash
cd "C:/Users/Макар/PycharmProjects/trading/CryptoTrade/ai/DRL"
.\venv\Scripts\Activate.ps1
```

### 2. Установка зависимостей (если нужно)

```bash
pip install gymnasium stable-baselines3 pandas numpy matplotlib plotly streamlit ta
```

### 3. Простейший пример

```python
from Enviroment import CryptoTradingEnv

# Создаем окружение
env = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    initial_balance=10000.0
)

# Используем
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Случайное действие
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

# Получаем метрики
metrics = env.get_metrics()
print(f"Return: {metrics.total_return_pct:.2f}%")
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
```

### 4. Обучение с Stable-Baselines3

```python
from stable_baselines3 import PPO
from Enviroment import CryptoTradingEnv

# Окружение
env = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    initial_balance=10000.0
)

# Модель
model = PPO("MlpPolicy", env, verbose=1)

# Обучение
model.learn(total_timesteps=50000)

# Сохранение
model.save("crypto_bot")
```

---

## 🏗 Архитектура

```
Enviroment/
├── env.py                    # Основное окружение (CryptoTradingEnv)
├── data_loader.py            # Загрузка и обработка данных
├── simulator.py              # Симуляция рынка (ордера, комиссии, слипедж)
├── metrics.py                # Расчет метрик производительности
├── visualization.py          # Графики и визуализация
├── dashboard.py              # Streamlit интерактивная панель
├── config.yaml               # Конфигурация (YAML)
├── trading_config.json       # Конфигурация (JSON)
├── test_environment.py       # Unit-тесты
├── example_usage.py          # Примеры использования
├── training_notebook.ipynb   # Jupyter notebook для обучения
├── __init__.py               # Экспорты модулей
└── README.md                 # Эта документация
```

### Основные классы

| Класс | Описание |
|-------|----------|
| `CryptoTradingEnv` | Главное торговое окружение (Gymnasium-совместимое) |
| `DataLoader` | Загрузка данных из CSV/Parquet, добавление индикаторов |
| `MarketSimulator` | Симуляция исполнения ордеров с реализмом |
| `MetricsCalculator` | Расчет торговых метрик |
| `TradingVisualizer` | Создание графиков и визуализаций |
| `TradingDashboard` | Streamlit интерактивная панель |

---

## 📖 Использование

### Создание окружения

```python
from Enviroment import CryptoTradingEnv, ActionSpace, RewardType

env = CryptoTradingEnv(
    # Данные
    symbol="BTCUSDT",           # Торговая пара
    timeframe="1d",              # Таймфрейм
    start_index=0,               # Начальный индекс
    end_index=None,              # Конечный индекс (None = все данные)

    # Торговля
    initial_balance=10000.0,     # Начальный баланс
    max_position_size=1.0,       # Максимальный размер позиции (доля)

    # Action space
    action_type=ActionSpace.DISCRETE,  # DISCRETE или CONTINUOUS

    # Observation
    observation_window=50,       # Размер окна истории
    add_indicators=True,         # Добавить технические индикаторы
    normalize_observations=True, # Нормализовать

    # Reward
    reward_type=RewardType.RISK_ADJUSTED,  # Тип награды
    reward_scaling=1.0,          # Масштаб
    turnover_penalty=0.0001,     # Штраф за оборот
    drawdown_penalty=0.001,      # Штраф за просадку

    # Симуляция рынка
    maker_fee=0.0001,            # Комиссия мейкера
    taker_fee=0.001,             # Комиссия тейкера
    slippage_percentage=0.0005,  # Проскальзывание

    # Завершение эпизода
    max_steps=None,              # Максимум шагов
    stop_on_bankruptcy=True,     # Остановить при банкротстве
)
```

### Action Spaces

**Дискретный (по умолчанию)**:
- `0` - Hold (держать)
- `1` - Buy (купить на весь баланс)
- `2` - Sell (продать всю позицию)

**Непрерывный**:
- Значение в `[-1, 1]`
- `-1` = продать всё
- `0` = держать
- `1` = купить на весь баланс

### Reward Types

| Тип | Описание |
|-----|----------|
| `PNL` | Чистая прибыль/убыток |
| `LOG_RETURN` | Логарифмическая доходность |
| `SHARPE` | Sharpe-подобная метрика |
| `SORTINO` | Sortino-подобная метрика |
| `RISK_ADJUSTED` | PnL - λ·turnover - μ·drawdown |

### Observation Space

Observation включает:
1. **Historical window** (window_size × features) - история цен и индикаторов
2. **Portfolio state** (3) - баланс, крипта, общая стоимость
3. **Position info** (2) - доля позиции, unrealized PnL
4. **Episode info** (2) - прогресс эпизода, флаг позиции

Всего: `window_size × n_features + 7`

---

## ⚙️ Конфигурация

### Через YAML (config.yaml)

```yaml
data:
  symbol: "BTCUSDT"
  timeframe: "1d"

trading:
  initial_balance: 10000.0

reward:
  type: "risk_adjusted"
  turnover_penalty: 0.0001
  drawdown_penalty: 0.001

market_simulation:
  maker_fee: 0.0001
  taker_fee: 0.001
  slippage_percentage: 0.0005
```

### Через JSON (trading_config.json)

```json
{
  "environment": {
    "name": "CryptoTradingEnv",
    "version": "1.0.0"
  },
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "1d"
  },
  "trading": {
    "initial_balance": 10000.0
  }
}
```

---

## 🎨 Интерактивная панель

### Запуск Dashboard

```bash
cd Enviroment
streamlit run dashboard.py
```

### Возможности:
- ⚙️ **Конфигурация окружения** через GUI
- 🎮 **Ручная торговля** (Buy/Hold/Sell кнопки)
- 🤖 **Автоматическая торговля** (случайные действия)
- 📊 **Real-time графики** (equity curve, rewards)
- 📝 **Лог сделок** с экспортом в CSV
- 📉 **Метрики производительности**
- 🔧 **Debug информация**

---

## 🧠 Обучение агента

### Базовый PPO

```python
from stable_baselines3 import PPO
from Enviroment import CryptoTradingEnv

# Train окружение
env = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    end_index=int(2855 * 0.8)  # 80% для обучения
)

# Модель
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# Обучение
model.learn(total_timesteps=100000)
model.save("crypto_ppo")

# Тестирование
test_env = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    start_index=int(2855 * 0.8)  # 20% для теста
)

obs, _ = test_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        break

metrics = test_env.get_metrics()
print(f"Test Return: {metrics.total_return_pct:.2f}%")
```

### Другие алгоритмы

```python
from stable_baselines3 import A2C, SAC

# A2C
model = A2C("MlpPolicy", env, verbose=1)

# SAC (требует continuous action space)
env_continuous = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    action_type=ActionSpace.CONTINUOUS
)
model = SAC("MlpPolicy", env_continuous, verbose=1)
```

---

## 📚 API Reference

### CryptoTradingEnv

```python
class CryptoTradingEnv(gym.Env):
    def reset(seed, options) -> Tuple[np.ndarray, Dict]
    def step(action) -> Tuple[np.ndarray, float, bool, bool, Dict]
    def render()
    def close()
    def get_metrics() -> PerformanceMetrics
    def seed(seed: int)
```

### DataLoader

```python
class DataLoader:
    def load(start_index, end_index) -> pd.DataFrame
    def get_window(start_idx, window_size) -> np.ndarray
    def get_price_at(idx, price_type) -> float
    def split_train_test(train_ratio) -> Tuple[DataLoader, DataLoader]
```

### MarketSimulator

```python
class MarketSimulator:
    def get_market_state(mid_price, volume, volatility) -> MarketState
    def execute_order(side, quantity, market_state) -> OrderResult
    def get_statistics() -> Dict
    def reset_history()
```

### MetricsCalculator

```python
class MetricsCalculator:
    def calculate_metrics(equity_curve, trades, timestamps) -> PerformanceMetrics
    def calculate_rolling_sharpe(equity_curve, window) -> np.ndarray
    def compare_with_baseline(strategy_metrics, baseline_equity) -> Dict
```

---

## 💡 Примеры

### 1. Buy & Hold Strategy

```python
env = CryptoTradingEnv(symbol="BTCUSDT", timeframe="1d")
obs, info = env.reset()

# Купить и держать
env.step(1)  # Buy
for _ in range(100):
    env.step(0)  # Hold

metrics = env.get_metrics()
print(f"Return: {metrics.total_return_pct:.2f}%")
```

### 2. Simple Momentum

```python
env = CryptoTradingEnv(symbol="BTCUSDT", timeframe="1d")
obs, info = env.reset()

price_history = []
for _ in range(150):
    current_price = info['current_price']
    price_history.append(current_price)

    if len(price_history) >= 10:
        momentum = (price_history[-1] - price_history[-10]) / price_history[-10]

        if momentum > 0.02 and env.crypto_held == 0:
            action = 1  # Buy
        elif momentum < -0.02 and env.crypto_held > 0:
            action = 2  # Sell
        else:
            action = 0  # Hold
    else:
        action = 0

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 3. Визуализация

```python
from Enviroment import TradingVisualizer

# После торговли
viz = TradingVisualizer()

viz.plot_full_analysis(
    data=env.data_loader.raw_data,
    equity_curve=env.equity_curve,
    trades=env.trades_history,
    metrics=env.get_metrics(),
    symbol="BTCUSDT",
    save_path="results.png"
)

# Интерактивный график
viz.create_interactive_plotly(
    data=env.data_loader.raw_data,
    equity_curve=env.equity_curve,
    trades=env.trades_history,
    metrics=env.get_metrics(),
    save_path="results.html"
)
```

---

## 🧪 Тестирование

### Запуск всех тестов

```bash
python test_environment.py
```

### Или с pytest

```bash
pytest test_environment.py -v
```

### Покрытие тестами

- ✅ DataLoader (загрузка, индикаторы, нормализация, splits)
- ✅ MarketSimulator (ордера, комиссии, слипедж)
- ✅ MetricsCalculator (Sharpe, drawdown, trade metrics)
- ✅ CryptoTradingEnv (reset, step, actions, rewards)

---

## 📊 Структура данных

### Требуемый формат данных

```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,50000.0,51000.0,49500.0,50500.0,1500.5
2024-01-02 00:00:00,50500.0,51500.0,50000.0,51200.0,1800.3
...
```

### Путь к данным

Автоматический поиск:
```
EnviromentData/Date/binance/{SYMBOL}/{TIMEFRAME}/{SYMBOL}_{TIMEFRAME}.parquet
```

Пример:
```
EnviromentData/Date/binance/BTCUSDT/1d/BTCUSDT_1d.parquet
```

---

## 🛡️ Предупреждение

⚠️ **ВАЖНО**: Это симуляционное окружение для образовательных и исследовательских целей.

- ❌ НЕ используйте непроверенные стратегии на реальных средствах
- ❌ Реальные рынки имеют дополнительные риски и сложности
- ❌ Прошлая производительность не гарантирует будущих результатов
- ✅ Всегда тщательно тестируйте на исторических данных
- ✅ Используйте управление рисками
- ✅ Начинайте с малых сумм при переходе на реальную торговлю

---

## 🤝 Вклад

Если вы хотите улучшить этот проект:

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

---

## 📝 Лицензия

MIT License - используйте свободно для обучения и исследований.

---

## 🙏 Благодарности

- **Gymnasium** - за отличный API окружений
- **Stable-Baselines3** - за готовые DRL алгоритмы
- **Binance** - за доступ к историческим данным
- **TA-Lib / TA** - за технические индикаторы

---

## 📧 Контакты

Вопросы? Проблемы? Предложения?

- 📁 Создайте Issue в репозитории
- 💬 Обсудите в Discussions

---

**Happy Trading! 🚀📈**
