# Deep Reinforcement Learning для торговли криптовалютой

## Обзор
Эта система реализует комплексное решение для алгоритмической торговли криптовалютой с использованием глубокого обучения с подкреплением (DRL).

## Архитектура системы

### 1. Компоненты системы
- **data_processing/**: Сбор и предобработка данных
- **environment/**: Среда симуляции торговли
- **agents/**: DRL алгоритмы (PPO, A2C, DDPG, DQN)
- **evaluation/**: Бэктестинг и оценка производительности
- **deployment/**: Развертывание в продакшн
- **utils/**: Вспомогательные утилиты

### 2. Поддерживаемые алгоритмы
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic) 
- DDPG (Deep Deterministic Policy Gradient)
- Double DQN (Double Deep Q-Network)
- Dueling Network Architectures

### 3. Источники данных
- CoinDesk Data API
- CoinMarketCap API
- CoinGecko API
- CCXT (множественные биржи)

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Сбор данных
python data_processing/data_collector.py

# Обучение агента
python training/train_agent.py --algorithm PPO --symbol BTCUSDT

# Оценка производительности
python evaluation/backtest.py --model_path models/ppo_model.zip
```

## Структура проекта

```
DRL/
├── data_processing/        # Сбор и предобработка данных
├── environment/           # Торговая среда симуляции
├── agents/               # DRL алгоритмы
├── training/             # Обучение моделей
├── evaluation/           # Оценка и бэктестинг
├── deployment/           # Продакшн развертывание
├── utils/               # Утилиты
├── config/              # Конфигурации
├── models/              # Сохраненные модели
├── logs/                # Логи обучения
└── requirements.txt     # Зависимости
```