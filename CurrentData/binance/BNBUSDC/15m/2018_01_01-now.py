import time
import psycopg2
import pandas as pd
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

# Настройки Binance API (можно оставить пустыми)
client = Client(api_key='', api_secret='')

# Настройки PostgreSQL
postgres_config = {
    'dbname': 'trading',
    'user': 'postgres',
    'password': '19856Roma',
    'host': 'localhost',
    'port': 5432
}

symbol = 'BNBUSDC'
interval = Client.KLINE_INTERVAL_15MINUTE
table_name = 'bnbusdc_15m'

# Создание таблицы
def create_table():
    with psycopg2.connect(**postgres_config) as conn:
        with conn.cursor() as cur:
            cur.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMPTZ PRIMARY KEY,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume NUMERIC
                );
            ''')
            conn.commit()

# Получение последнего времени в базе
def get_last_timestamp():
    with psycopg2.connect(**postgres_config) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(timestamp) FROM {table_name};")
            result = cur.fetchone()[0]
            return result

# Сохранение данных в PostgreSQL
def save_to_postgres(klines):
    with psycopg2.connect(**postgres_config) as conn:
        with conn.cursor() as cur:
            for k in klines:
                ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                cur.execute(f'''
                    INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                ''', (ts, k[1], k[2], k[3], k[4], k[5]))
            conn.commit()

# Загрузка исторических данных с 2018 года
def load_history():
    print("Загрузка исторических данных...")
    start_date = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    delta = timedelta(days=10)

    while start_date < end_date:
        next_date = min(start_date + delta, end_date)
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date.isoformat(),
                end_str=next_date.isoformat(),
                limit=1000
            )
            if not klines:
                break

            # Преобразуем строки в числа
            for k in klines:
                k[1:6] = list(map(float, k[1:6]))

            save_to_postgres(klines)
            print(f"Загружено: {start_date} — {next_date}")
            start_date = next_date
            time.sleep(1)  # не нагружаем API
        except BinanceAPIException as e:
            print(f"Ошибка Binance API: {e}")
            time.sleep(5)

# Загрузка данных из БД в DataFrame
def load_data():
    with psycopg2.connect(**postgres_config) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

# Обновление данных в реальном времени
def live_update():
    last_ts = get_last_timestamp()
    if not last_ts:
        return

    since = int(last_ts.timestamp() * 1000)
    klines = client.get_klines(symbol=symbol, interval=interval, startTime=since, limit=2)

    if klines:
        for k in klines:
            k[1:6] = list(map(float, k[1:6]))
        save_to_postgres(klines)
        print(f"Обновлено: {datetime.utcnow().isoformat()} ({len(klines)} свечей)")

# Отрисовка графика
def plot_graph(df):
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="BNB/USDC — 15m")
    win.resize(1000, 600)
    plot = win.addPlot()
    plot.setTitle("Свечной график BNBUSDC (15 минут)")
    plot.showGrid(x=True, y=True)

    # Создание свечей
    candles = []
    for i in range(len(df)):
        ts = df['timestamp'].iloc[i].timestamp()
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
        candle = {
            'timestamp': ts,
            'open': o,
            'high': h,
            'low': l,
            'close': c
        }
        candles.append(candle)

    # Отображение свечей
    for i, c in enumerate(candles):
        color = 'g' if c['close'] >= c['open'] else 'r'
        line = pg.QtGui.QGraphicsLineItem(i, c['low'], i, c['high'])
        line.setPen(pg.mkPen(color))
        rect = pg.QtGui.QGraphicsRectItem(i - 0.3, min(c['open'], c['close']),
                                          0.6, abs(c['close'] - c['open']))
        rect.setPen(pg.mkPen(color))
        rect.setBrush(pg.mkBrush(color))
        plot.addItem(line)
        plot.addItem(rect)

    timer = QtCore.QTimer()
    def refresh():
        live_update()
        new_df = load_data()
        # можно оптимизировать перерисовку только последних свечей

    timer.timeout.connect(refresh)
    timer.start(60000)  # каждые 60 секунд
    QtGui.QApplication.instance().exec_()

# === ЗАПУСК ===
if __name__ == "__main__":
    create_table()
    load_history()
    df = load_data()
    print("Запуск обновления в реальном времени...")
    plot_graph(df)
