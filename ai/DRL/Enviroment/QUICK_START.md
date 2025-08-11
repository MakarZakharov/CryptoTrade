# 🚀 Быстрый старт

## Шаг 1: Активация виртуального окружения

```bash
cd "C:/Users/Макар/PycharmProjects/trading/CryptoTrade/ai/DRL"
.\venv\Scripts\Activate.ps1
```

## Шаг 2: Проверка установки

Все необходимые библиотеки уже установлены:
- ✅ gymnasium
- ✅ stable-baselines3
- ✅ pandas, numpy
- ✅ matplotlib, plotly
- ✅ streamlit
- ✅ ta (технические индикаторы)

## Шаг 3: Запуск примеров

### Пример 1: Базовое использование
```bash
cd Enviroment
python example_usage.py
```

### Пример 2: Запуск тестов
```bash
python test_environment.py
```

### Пример 3: Интерактивная панель
```bash
streamlit run dashboard.py
```

### Пример 4: Jupyter Notebook
```bash
jupyter notebook training_notebook.ipynb
```

## Шаг 4: Первый код

```python
from Enviroment import CryptoTradingEnv

# Создаем окружение
env = CryptoTradingEnv(
    symbol="BTCUSDT",
    timeframe="1d",
    initial_balance=10000.0
)

# Запускаем
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

# Смотрим результаты
metrics = env.get_metrics()
print(f"Return: {metrics.total_return_pct:.2f}%")
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
```

## Шаг 5: Обучение DRL агента

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
model.save("my_crypto_bot")
```

## 📚 Дальнейшие шаги

1. Изучите [README.md](README.md) для полной документации
2. Откройте [training_notebook.ipynb](training_notebook.ipynb) для пошагового обучения
3. Экспериментируйте с параметрами в [config.yaml](config.yaml)
4. Запустите dashboard для интерактивного тестирования

## 🆘 Помощь

Если что-то не работает:
1. Проверьте активацию виртуального окружения
2. Убедитесь, что данные загружены в `EnviromentData/Date/binance/`
3. Запустите тесты: `python test_environment.py`

**Удачи! 🎉**
