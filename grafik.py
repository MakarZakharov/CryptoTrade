import sys
import asyncio
import json
import requests
import datetime
import pandas as pd
import numpy as np
import websockets
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

symbol = "BTCUSDT"
interval = "5m"
limit = 100


def fetch_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        '_', '_', '_', '_', '_', '_'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['timestamp'] = df['time'].astype(np.int64) // 10**9  # в секундах
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})
    return df


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)
        for idx, row in self.data.iterrows():
            t = row['timestamp']
            open_ = row['open']
            close = row['close']
            high = row['high']
            low = row['low']
            color = pg.mkColor('g') if close >= open_ else pg.mkColor('r')
            painter.setPen(pg.mkPen(color))
            # линия high-low
            painter.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            # тело свечи
            painter.drawRect(QtCore.QRectF(t - 60, open_, 120, close - open_))
        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        times = self.data['timestamp']
        return QtCore.QRectF(times.min(), self.data['low'].min(),
                             times.max() - times.min(), self.data['high'].max() - self.data['low'].min())


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.datetime.fromtimestamp(v).strftime("%H:%M") for v in values]


class RealTimeChart(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Chart - PyQtGraph")

        self.df = fetch_historical_data()

        self.plot_widget = pg.PlotWidget(axisItems={'bottom': TimeAxis(orientation='bottom')})
        self.setCentralWidget(self.plot_widget)

        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('k')
        self.plot_widget.setMouseEnabled(x=True, y=True)

        self.candles = CandlestickItem(self.df)
        self.plot_widget.addItem(self.candles)

        self.loop = asyncio.get_event_loop()
        asyncio.ensure_future(self.update_realtime())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw)
        self.timer.start(5000)

    def redraw(self):
        self.plot_widget.clear()
        self.candles = CandlestickItem(self.df)
        self.plot_widget.addItem(self.candles)

    async def update_realtime(self):
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@miniTicker"
        async with websockets.connect(url) as ws:
            while True:
                data = json.loads(await ws.recv())
                price = float(data['c'])

                # Обновляем последнюю свечу
                self.df.at[self.df.index[-1], 'high'] = max(self.df.iloc[-1]['high'], price)
                self.df.at[self.df.index[-1], 'low'] = min(self.df.iloc[-1]['low'], price)
                self.df.at[self.df.index[-1], 'close'] = price


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    chart = RealTimeChart()
    chart.show()
    sys.exit(app.exec_())
