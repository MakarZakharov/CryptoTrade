"""
Скрипт для сбора данных BTCUSDT с Binance во всех таймфреймах
Сохранение в формате Parquet с последующей визуализацией
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class BinanceDataCollector:
    """Класс для сбора и обработки данных с Binance"""

    def __init__(self, symbol: str = "BTCUSDT", base_path: str = None):
        self.symbol = symbol
        self.api_url = "https://api.binance.com/api/v3/klines"

        # Определяем базовый путь для сохранения данных
        if base_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_path = os.path.join(current_dir, "..", "..", "Date", "binance", symbol)
        else:
            self.base_path = base_path

        # Таймфреймы для сбора данных
        self.timeframes = ["15m", "1h", "4h", "1d"]

        # Начальная дата
        self.start_date = "2018-01-01"

    def get_klines(self, interval: str, start_ts: int, end_ts: int) -> List:
        """Получение свечных данных с Binance API"""
        all_klines = []
        limit = 1000
        current_start = start_ts

        print(f"   📥 Загрузка данных для {interval}...")

        while current_start < end_ts:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit
            }

            try:
                response = requests.get(self.api_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data or (isinstance(data, dict) and "code" in data):
                    print(f"   ⚠️ Ошибка в данных: {data}")
                    break

                all_klines += data
                current_start = data[-1][6] + 1  # Следующий запрос после последней свечи

                # Прогресс
                progress = min(100, int(((data[-1][0] - start_ts) / (end_ts - start_ts)) * 100))
                print(f"   ⏳ Прогресс: {progress}% ({len(all_klines)} свечей)", end='\r')

                if len(data) < limit:
                    break

                time.sleep(0.1)  # Задержка для избежания rate limit

            except requests.exceptions.RequestException as e:
                print(f"   ❌ Ошибка запроса: {e}")
                time.sleep(1)
                continue

        print(f"   ✅ Загружено {len(all_klines)} свечей для {interval}        ")
        return all_klines

    def klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Преобразование свечных данных в DataFrame"""
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])

        # Преобразование типов
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.astype({
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "quote_volume": "float",
            "num_trades": "int",
        })

        # Оставляем только нужные колонки
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "num_trades"]]

        return df

    def save_to_parquet(self, df: pd.DataFrame, interval: str) -> str:
        """Сохранение DataFrame в формат Parquet"""
        # Создаем директорию если не существует
        save_dir = os.path.join(self.base_path, interval)
        os.makedirs(save_dir, exist_ok=True)

        # Путь к файлу parquet
        filename = f"2018_01_01-{datetime.now().strftime('%Y_%m_%d')}.parquet"
        filepath = os.path.join(save_dir, filename)

        # Сохраняем с компрессией
        df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=False)

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Размер в МБ
        print(f"   💾 Сохранено: {filepath}")
        print(f"   📊 Размер файла: {file_size:.2f} МБ")
        print(f"   📈 Записей: {len(df):,}")

        return filepath

    def collect_all_timeframes(self) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """Сбор данных для всех таймфреймов"""
        results = {}

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.now()

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        print(f"\n{'='*60}")
        print(f"🚀 Начало сбора данных для {self.symbol}")
        print(f"📅 Период: {self.start_date} - {end_date.strftime('%Y-%m-%d')}")
        print(f"⏱️  Таймфреймы: {', '.join(self.timeframes)}")
        print(f"{'='*60}\n")

        for interval in self.timeframes:
            print(f"\n📊 Обработка таймфрейма: {interval}")
            print(f"{'-'*60}")

            try:
                # Получаем данные
                klines = self.get_klines(interval, start_ts, end_ts)

                if not klines:
                    print(f"   ⚠️ Нет данных для {interval}")
                    continue

                # Преобразуем в DataFrame
                df = self.klines_to_dataframe(klines)

                # Сохраняем в Parquet
                filepath = self.save_to_parquet(df, interval)

                # Сохраняем результат
                results[interval] = (df, filepath)

                print(f"   ✅ Таймфрейм {interval} обработан успешно!")

            except Exception as e:
                print(f"   ❌ Ошибка при обработке {interval}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"✅ Сбор данных завершен!")
        print(f"📁 Обработано таймфреймов: {len(results)}/{len(self.timeframes)}")
        print(f"{'='*60}\n")

        return results

    def plot_all_data(self, results: Dict[str, Tuple[pd.DataFrame, str]]):
        """Построение графиков для всех собранных данных"""
        print(f"\n{'='*60}")
        print(f"📈 Построение графиков для {self.symbol}")
        print(f"{'='*60}\n")

        # Создаем фигуру с подграфиками
        n_timeframes = len(results)
        fig, axes = plt.subplots(n_timeframes, 1, figsize=(16, 5 * n_timeframes))

        if n_timeframes == 1:
            axes = [axes]

        for idx, (interval, (df, filepath)) in enumerate(sorted(results.items())):
            print(f"   📊 График для {interval}...")

            ax = axes[idx]

            # Подготовка данных для графика
            df_plot = df.copy()
            df_plot.set_index('timestamp', inplace=True)

            # Используем все свечи с 2018 года
            # df_plot = df_plot.tail(500)  # Закомментировано - показываем все данные

            # Строим график цены
            ax.plot(df_plot.index, df_plot['close'], label='Close Price', linewidth=1.5, color='#2E86AB')
            ax.fill_between(df_plot.index, df_plot['low'], df_plot['high'], alpha=0.2, color='#A23B72')

            # Настройка графика
            ax.set_title(f'{self.symbol} - {interval.upper()} (все данные: {len(df_plot):,} свечей)',
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Дата', fontsize=10)
            ax.set_ylabel('Цена (USDT)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left')

            # Добавляем статистику
            stats_text = (
                f"Min: ${df_plot['low'].min():,.2f}\n"
                f"Max: ${df_plot['high'].max():,.2f}\n"
                f"Avg: ${df_plot['close'].mean():,.2f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

            # Форматирование оси X
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Сохраняем график
        chart_path = os.path.join(self.base_path, f"{self.symbol}_all_timeframes.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"\n   💾 График сохранен: {chart_path}")

        # Показываем график
        plt.show()

        print(f"   ✅ Визуализация завершена!\n")

    def plot_candlestick_charts(self, results: Dict[str, Tuple[pd.DataFrame, str]]):
        """Построение свечных графиков для всех таймфреймов в высоком качестве"""
        print(f"\n{'='*60}")
        print(f"🕯️  Построение свечных графиков в высоком качестве")
        print(f"{'='*60}\n")

        for interval, (df, filepath) in sorted(results.items()):
            try:
                print(f"   📊 Свечной график для {interval}...")

                # Подготовка данных
                df_plot = df.copy()
                df_plot.set_index('timestamp', inplace=True)
                df_plot = df_plot[["open", "high", "low", "close", "volume"]]

                # Используем все свечи с 2018 года
                # df_plot = df_plot.tail(200)  # Закомментировано - показываем все данные

                # Настройки стиля для высокого качества
                mc = mpf.make_marketcolors(
                    up='#26a69a',      # Зеленый для роста
                    down='#ef5350',    # Красный для падения
                    edge='inherit',
                    wick={'up':'#26a69a', 'down':'#ef5350'},
                    volume='in',
                    ohlc='inherit'
                )

                s = mpf.make_mpf_style(
                    marketcolors=mc,
                    gridstyle=':',
                    gridcolor='#e0e0e0',
                    facecolor='#ffffff',
                    edgecolor='#cccccc',
                    figcolor='#ffffff',
                    rc={'font.size': 10}
                )

                # Построение графика
                total_candles = len(df_plot)
                title = f"{self.symbol} - {interval.upper()} (всего: {total_candles:,} свечей)\n{self.start_date} - {datetime.now().strftime('%Y-%m-%d')}"

                # Путь сохранения в папке с данными
                save_path = os.path.join(self.base_path, interval, f"{self.symbol}_{interval}_candlestick_HQ.png")

                # Высокое качество графика
                savefig_config = dict(
                    fname=save_path,
                    dpi=300,  # Высокое разрешение
                    bbox_inches='tight',
                    pad_inches=0.2,
                    facecolor='white',
                    edgecolor='none'
                )

                # Построение с оптимальным размером
                mpf.plot(
                    df_plot,
                    type='candle',
                    style=s,
                    volume=True,
                    title=title,
                    ylabel='Цена (USDT)',
                    ylabel_lower='Объем',
                    savefig=savefig_config,
                    figsize=(20, 12),  # Больший размер для лучшей читаемости
                    tight_layout=True
                )

                # Проверяем что файл создался
                if os.path.exists(save_path):
                    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"   ✅ Сохранено: {save_path}")
                    print(f"      Размер: {file_size_mb:.2f} МБ, Свечей: {total_candles:,}")
                else:
                    print(f"   ⚠️ Файл не создан: {save_path}")

            except Exception as e:
                print(f"   ❌ Ошибка построения графика для {interval}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n   ✅ Все свечные графики построены!\n")

    def generate_summary_report(self, results: Dict[str, Tuple[pd.DataFrame, str]]):
        """Генерация сводного отчета"""
        print(f"\n{'='*60}")
        print(f"📋 СВОДНЫЙ ОТЧЕТ")
        print(f"{'='*60}\n")

        print(f"🪙 Символ: {self.symbol}")
        print(f"📅 Период: {self.start_date} - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"📁 Путь сохранения: {self.base_path}\n")

        print(f"{'Таймфрейм':<12} {'Свечей':<15} {'Размер файла':<20} {'Период данных'}")
        print(f"{'-'*80}")

        for interval, (df, filepath) in sorted(results.items()):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
            end_date = df['timestamp'].max().strftime('%Y-%m-%d')

            print(f"{interval:<12} {len(df):>10,}     {file_size:>8.2f} MB      {start_date} - {end_date}")

        print(f"\n{'='*60}\n")


def main():
    """Основная функция запуска"""
    print("\n" + "="*60)
    print(" "*15 + "🚀 BINANCE DATA COLLECTOR")
    print(" "*20 + "Parquet Edition")
    print("="*60 + "\n")

    # Создаем экземпляр коллектора
    collector = BinanceDataCollector(symbol="BTCUSDT")

    # Собираем данные для всех таймфреймов
    results = collector.collect_all_timeframes()

    if not results:
        print("❌ Не удалось собрать данные!")
        return

    # Генерируем сводный отчет
    collector.generate_summary_report(results)

    # Строим графики
    collector.plot_all_data(results)

    # Строим свечные графики
    collector.plot_candlestick_charts(results)

    print("\n" + "="*60)
    print(" "*20 + "✅ ГОТОВО!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
