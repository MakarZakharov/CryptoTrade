# Система аналізу криптовалютного ринку

Ця система дозволяє аналізувати дані криптовалютного ринку, навчати моделі машинного навчання для прогнозування цін та використовувати ці моделі для передбачення майбутніх цін.

## Навчання моделі

Для навчання моделі використовуйте скрипт `main.py`. Ось приклади використання:

### Базове навчання LSTM моделі для BTC/USDT

```bash
python3 main.py --symbol BTCUSDT --timeframe 1d --start_date 2020-01-01 --model_type lstm --save_model models/btc_lstm
```

### Навчання GRU моделі для ETH/USDT

```bash
python3 main.py --symbol ETHUSDT --timeframe 1d --start_date 2020-01-01 --model_type gru --save_model models/eth_gru
```

### Навчання ансамблевої моделі

```bash
python3 main.py --symbol BTCUSDT --model_type stacking --ensemble_models lstm,xgboost --save_model models/btc_ensemble
```

### Параметри командного рядка

- `--symbol`: Торгова пара (наприклад, BTCUSDT)
- `--timeframe`: Часовий інтервал (1d, 4h, 1h)
- `--start_date`: Початкова дата для даних
- `--end_date`: Кінцева дата для даних (за замовчуванням - поточна дата)
- `--data_source`: Джерело даних (binance або csv)
- `--csv_path`: Шлях до CSV файлу (якщо data_source=csv)
- `--window_size`: Розмір вікна для послідовних даних
- `--train_split`: Частка даних для навчання
- `--val_split`: Частка даних для валідації
- `--model_type`: Тип моделі (lstm, gru, transformer, xgboost, stacking, voting)
- `--ensemble_models`: Список моделей для ансамблю (розділені комами)
- `--units`: Кількість юнітів у LSTM/GRU шарах
- `--dropout`: Коефіцієнт dropout
- `--learning_rate`: Швидкість навчання
- `--batch_size`: Розмір батчу для навчання
- `--epochs`: Кількість епох навчання
- `--early_stopping`: Терпіння для раннього зупинення
- `--save_model`: Шлях для збереження моделі
- `--load_model`: Шлях для завантаження моделі
- `--no_train`: Пропустити навчання (використовувати з --load_model)
- `--no_plot`: Пропустити візуалізацію результатів

## Прогнозування на реальних даних

Для прогнозування на реальних даних використовуйте скрипт `predict.py`. Цей скрипт завантажує навчену модель і використовує її для прогнозування майбутніх цін.

### Приклади використання

#### Прогнозування цін BTC на наступні 7 днів

```bash
python3 predict.py --symbol BTCUSDT --model_path models/btc_lstm --model_type lstm --forecast_days 7
```

#### Прогнозування цін ETH на наступні 14 днів

```bash
python3 predict.py --symbol ETHUSDT --model_path models/eth_gru --model_type gru --forecast_days 14
```

#### Збереження прогнозів у файл

```bash
python3 predict.py --symbol BTCUSDT --model_path models/btc_lstm --model_type lstm --output_file predictions/btc_pred.csv
```

### Параметри командного рядка

- `--symbol`: Торгова пара (наприклад, BTCUSDT)
- `--timeframe`: Часовий інтервал (1d, 4h, 1h)
- `--days`: Кількість днів історичних даних для завантаження
- `--model_path`: Шлях до навченої моделі (обов'язковий)
- `--model_type`: Тип моделі (lstm, gru, transformer, xgboost, stacking, voting)
- `--forecast_days`: Кількість днів для прогнозування
- `--output_file`: Шлях для збереження прогнозів
- `--no_plot`: Пропустити візуалізацію результатів

## Структура проекту

- `data/`: Модулі для завантаження та обробки даних
  - `fetchers/`: Класи для завантаження даних з різних джерел
  - `processors/`: Класи для обробки даних
  - `features/`: Класи для інженерії ознак
- `models/`: Модулі для моделей машинного навчання
- `plots/`: Згенеровані візуалізації
- `models/`: Збережені моделі
- `main.py`: Головний скрипт для навчання моделей
- `predict.py`: Скрипт для прогнозування на реальних даних

## Вимоги

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- xgboost
- requests

## Встановлення

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib xgboost requests