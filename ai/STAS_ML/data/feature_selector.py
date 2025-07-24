"""
Автоматический селектор признаков/индикаторов для STAS_ML.
Выбирает оптимальные технические индикаторы для каждой модели.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class AutomaticFeatureSelector:
    """Автоматический селектор лучших технических индикаторов."""
    
    def __init__(self, config):
        self.config = config
        self.selected_indicators = {}
        self.indicator_scores = {}
        self.feature_importance = {}
        
        # Полный список доступных индикаторов
        self.all_indicators = {
            'trend': {
                'sma': [5, 10, 20, 50, 200],
                'ema': [5, 10, 20, 50, 200], 
                'macd': [12, 26, 9],
                'adx': [14],
                'aroon': [14],
                'cci': [14, 20],
                'dx': [14]
            },
            'momentum': {
                'rsi': [6, 14, 21],
                'stochastic': [14, 3, 3],
                'williams_r': [14],
                'roc': [10, 12],
                'momentum': [10],
                'trix': [14]
            },
            'volatility': {
                'bollinger': [20],
                'atr': [14, 21, 28],
                'natr': [14],
                'keltner': [20],
                'donchian': [20]
            },
            'volume': {
                'obv': [],
                'ad': [],
                'chaikin_ad': [3, 10],
                'mfi': [14],
                'volume_sma': [20],
                'vwap': []
            },
            'overlap': {
                'midpoint': [14],
                'midprice': [14],
                'sar': [0.02, 0.2],
                'tema': [30],
                'trima': [30]
            }
        }
    
    def select_best_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Автоматически выбрать лучшие индикаторы для данной торговой пары.
        
        Returns:
            Dict с выбранными индикаторами и их оценками
        """
        print(f"🔍 Автоматический выбор индикаторов для {self.config.symbol}...")
        
        # Этап 1: Генерируем все возможные индикаторы
        all_indicators_data = self._generate_all_indicators(data)
        
        # Этап 2: Предварительная фильтрация по вариации
        filtered_data = self._filter_low_variance(all_indicators_data)
        
        # Этап 3: Выбор топ индикаторов по корреляции с целевой переменной
        correlation_selected = self._select_by_correlation(filtered_data, data)
        
        # Этап 4: Выбор финальных индикаторов используя ML методы
        final_indicators = self._select_by_ml_importance(correlation_selected, data)
        
        # Этап 5: Кросс-валидация выбранных индикаторов
        validated_indicators = self._cross_validate_indicators(final_indicators, data)
        
        print(f"✅ Выбрано {len(validated_indicators)} лучших индикаторов")
        self._print_selected_indicators(validated_indicators)
        
        return validated_indicators
    
    def _generate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерировать все возможные технические индикаторы."""
        df = data.copy()
        
        print("🔄 Генерируем все доступные индикаторы...")
        
        if TALIB_AVAILABLE:
            df = self._add_all_talib_indicators(df)
        else:
            df = self._add_all_pandas_indicators(df)
        
        # Базовые фичи
        df = self._add_basic_features(df)
        
        print(f"📊 Сгенерировано {len(df.columns) - len(data.columns)} индикаторов")
        
        return df
    
    def _add_all_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить все доступные TA-Lib индикаторы."""
        
        # Трендовые индикаторы
        for period in self.all_indicators['trend']['sma']:
            try:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            except:
                pass
                
        for period in self.all_indicators['trend']['ema']:
            try:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            except:
                pass
        
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal  
            df['macd_histogram'] = macd_hist
        except:
            pass
        
        # ADX
        try:
            df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        except:
            pass
        
        # Aroon
        try:
            aroon_down, aroon_up = talib.AROON(df['high'], df['low'], timeperiod=14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_osc'] = aroon_up - aroon_down
        except:
            pass
        
        # CCI
        for period in self.all_indicators['trend']['cci']:
            try:
                df[f'cci_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
            except:
                pass
        
        # Momentum индикаторы
        for period in self.all_indicators['momentum']['rsi']:
            try:
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            except:
                pass
        
        # Stochastic
        try:
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
        except:
            pass
        
        # Williams %R
        try:
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        except:
            pass
        
        # ROC
        for period in self.all_indicators['momentum']['roc']:
            try:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            except:
                pass
        
        # Momentum
        try:
            df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
        except:
            pass
        
        # TRIX
        try:
            df['trix'] = talib.TRIX(df['close'], timeperiod=14)
        except:
            pass
        
        # Volatility индикаторы
        for period in self.all_indicators['volatility']['bollinger']:
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=period)
                df[f'bb_upper_{period}'] = bb_upper
                df[f'bb_middle_{period}'] = bb_middle
                df[f'bb_lower_{period}'] = bb_lower
                df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                df[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            except:
                pass
        
        for period in self.all_indicators['volatility']['atr']:
            try:
                df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                df[f'natr_{period}'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=period)
            except:
                pass
        
        # Volume индикаторы
        try:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        except:
            pass
        
        try:
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        except:
            pass
        
        try:
            df['chaikin_ad'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        except:
            pass
        
        try:
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        except:
            pass
        
        # Overlap индикаторы
        try:
            df['midpoint'] = talib.MIDPOINT(df['close'], timeperiod=14)
        except:
            pass
        
        try:
            df['midprice'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=14)
        except:
            pass
        
        try:
            df['sar'] = talib.SAR(df['high'], df['low'])
        except:
            pass
        
        try:
            df['tema'] = talib.TEMA(df['close'], timeperiod=30)
        except:
            pass
        
        try:
            df['trima'] = talib.TRIMA(df['close'], timeperiod=30)
        except:
            pass
        
        return df
    
    def _add_all_pandas_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить базовые индикаторы используя pandas (fallback)."""
        
        # SMA
        for period in self.all_indicators['trend']['sma']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # EMA 
        for period in self.all_indicators['trend']['ema']:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        for period in self.all_indicators['momentum']['rsi']:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # ATR
        for period in self.all_indicators['volatility']['atr']:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить базовые фичи."""
        # Ценовые изменения
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Объемные изменения
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Ценовые соотношения
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        df['high_close_ratio'] = (df['high'] - df['close']) / df['close']
        df['low_close_ratio'] = (df['close'] - df['low']) / df['close']
        
        # Волатильность
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
            df[f'returns_skew_{window}'] = df['price_change'].rolling(window=window).skew()
            df[f'returns_kurt_{window}'] = df['price_change'].rolling(window=window).kurt()
        
        # Межвременные соотношения
        df['close_vs_sma_20'] = df['close'] / df['close'].rolling(20).mean()
        df['volume_vs_sma_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def _filter_low_variance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удалить признаки с низкой вариацией."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Фильтр вариации
        selector = VarianceThreshold(threshold=0.01)  # Минимальная вариация
        
        try:
            data_filtered = data[numeric_columns].dropna()
            selector.fit(data_filtered)
            
            selected_columns = numeric_columns[selector.get_support()]
            print(f"🔍 Отфильтровано {len(numeric_columns) - len(selected_columns)} признаков с низкой вариацией")
            
            # Возвращаем оригинальные колонки + отобранные
            original_columns = ['open', 'high', 'low', 'close', 'volume']
            result_columns = list(set(original_columns + list(selected_columns)))
            
            return data[result_columns]
        except:
            print("⚠️ Ошибка фильтрации по вариации, используем все признаки")
            return data[numeric_columns]
    
    def _select_by_correlation(self, data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Выбрать индикаторы по корреляции с ценовыми движениями."""
        # Создаем целевую переменную для корреляционного анализа
        target = original_data['close'].pct_change().shift(-1)  # Следующий возврат
        
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        # Рассчитываем корреляции
        correlations = {}
        for col in numeric_data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    corr = abs(numeric_data[col].corr(target))
                    if not np.isnan(corr):
                        correlations[col] = corr
                except:
                    pass
        
        # Выбираем топ корреляционные признаки
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:50]
        
        selected_columns = ['open', 'high', 'low', 'close', 'volume'] + [feat[0] for feat in top_features]
        
        print(f"🔍 Выбрано {len(top_features)} признаков по корреляции")
        
        return data[selected_columns]
    
    def _select_by_ml_importance(self, data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Выбрать финальные индикаторы используя ML важность признаков."""
        
        # Подготавливаем данные
        data_clean = data.dropna()
        
        if len(data_clean) < 100:
            print("⚠️ Недостаточно данных для ML селекции")
            return self._fallback_indicators()
        
        # Создаем цель
        if self.config.target_type == 'direction':
            target = (original_data['close'].pct_change().shift(-1) > 0).astype(int)
        else:
            target = original_data['close'].pct_change().shift(-1)
        
        target_clean = target.loc[data_clean.index].dropna()
        data_final = data_clean.loc[target_clean.index]
        
        if len(data_final) < 50:
            print("⚠️ Недостаточно данных после очистки")
            return self._fallback_indicators()
        
        # Выбираем только числовые признаки (исключая OHLCV)
        feature_columns = [col for col in data_final.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = data_final[feature_columns].values
        y = target_clean.values
        
        if len(feature_columns) == 0:
            return self._fallback_indicators()
        
        # Используем Random Forest для важности признаков
        try:
            if self.config.target_type == 'direction':
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            rf.fit(X, y)
            
            # Получаем важность признаков
            feature_importance = dict(zip(feature_columns, rf.feature_importances_))
            
            # Выбираем топ признаки
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            selected_indicators = {
                'selected_features': [feat[0] for feat in top_features],
                'feature_importance': dict(top_features),
                'selection_method': 'random_forest',
                'n_features': len(top_features)
            }
            
            print(f"🤖 ML выбрал {len(top_features)} лучших признаков")
            
            return selected_indicators
            
        except Exception as e:
            print(f"⚠️ Ошибка ML селекции: {e}")
            return self._fallback_indicators()
    
    def _cross_validate_indicators(self, indicators: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Кросс-валидация выбранных индикаторов."""
        
        if 'selected_features' not in indicators:
            return indicators
        
        # Простая валидация: проверяем что индикаторы не слишком коррелированы между собой
        selected_features = indicators['selected_features']
        
        try:
            # Берем данные по выбранным признакам
            data_selected = data[selected_features].dropna()
            
            if len(data_selected) < 10:
                return indicators
            
            # Корреляционная матрица
            corr_matrix = data_selected.corr().abs()
            
            # Удаляем слишком коррелированные признаки
            final_features = []
            for i, feature in enumerate(selected_features):
                # Проверяем корреляцию с уже добавленными признаками
                is_unique = True
                for existing_feature in final_features:
                    if feature in corr_matrix.columns and existing_feature in corr_matrix.columns:
                        if corr_matrix.loc[feature, existing_feature] > 0.9:
                            is_unique = False
                            break
                
                if is_unique:
                    final_features.append(feature)
                
                # Ограничиваем количество признаков
                if len(final_features) >= 15:
                    break
            
            indicators['selected_features'] = final_features
            indicators['n_features'] = len(final_features)
            
            print(f"✅ После кросс-валидации осталось {len(final_features)} уникальных признаков")
            
        except Exception as e:
            print(f"⚠️ Ошибка кросс-валидации: {e}")
        
        return indicators
    
    def _fallback_indicators(self) -> Dict[str, Any]:
        """Резервный набор индикаторов при ошибках."""
        fallback_features = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 
            'bb_upper_20', 'bb_lower_20', 'atr_14', 'obv',
            'price_change', 'volume_change', 'volatility_20'
        ]
        
        return {
            'selected_features': fallback_features,
            'feature_importance': {feat: 1.0 for feat in fallback_features},
            'selection_method': 'fallback',
            'n_features': len(fallback_features)
        }
    
    def _print_selected_indicators(self, indicators: Dict[str, Any]):
        """Вывести информацию о выбранных индикаторах."""
        print(f"\n📊 ВЫБРАННЫЕ ИНДИКАТОРЫ:")
        print(f"   Метод: {indicators.get('selection_method', 'unknown')}")
        print(f"   Количество: {indicators.get('n_features', 0)}")
        
        if 'feature_importance' in indicators:
            print(f"   Топ-5 по важности:")
            sorted_features = sorted(indicators['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"     {i+1}. {feature}: {importance:.4f}")
    
    def visualize_feature_importance(self, indicators: Dict[str, Any], save_path: str = None) -> go.Figure:
        """
        Создание интерактивного графика важности признаков.
        
        Args:
            indicators: Результаты селекции признаков
            save_path: Путь для сохранения графика
            
        Returns:
            Plotly Figure объект
        """
        if 'feature_importance' not in indicators:
            print("⚠️ Данные о важности признаков недоступны")
            return None
        
        # Подготовка данных
        feature_importance = indicators['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        features = [feat[0] for feat in sorted_features]
        importances = [feat[1] for feat in sorted_features]
        
        # Создание интерактивного графика
        fig = go.Figure()
        
        # Добавление столбчатой диаграммы
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Важность")
            ),
            text=[f"{imp:.4f}" for imp in importances],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Важность: %{x:.4f}<extra></extra>'
        ))
        
        # Настройка макета
        fig.update_layout(
            title=f'Важность признаков ({indicators.get("selection_method", "unknown")})',
            xaxis_title='Важность признака',
            yaxis_title='Технические индикаторы',
            height=max(400, len(features) * 25),
            showlegend=False,
            template='plotly_white',
            font=dict(size=12),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # Сохранение если указан путь
        if save_path:
            fig.write_html(save_path)
            print(f"📊 График важности сохранен: {save_path}")
        
        return fig
    
    def visualize_correlation_matrix(self, data: pd.DataFrame, selected_features: List[str], 
                                   save_path: str = None) -> go.Figure:
        """
        Создание интерактивной корреляционной матрицы выбранных признаков.
        
        Args:
            data: Данные с признаками
            selected_features: Список выбранных признаков
            save_path: Путь для сохранения
            
        Returns:
            Plotly Figure объект
        """
        try:
            # Фильтрация доступных признаков
            available_features = [feat for feat in selected_features if feat in data.columns]
            
            if len(available_features) < 2:
                print("⚠️ Недостаточно признаков для корреляционной матрицы")
                return None
            
            # Расчет корреляции
            corr_matrix = data[available_features].corr()
            
            # Создание тепловой карты
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Корреляция: %{z:.3f}<extra></extra>'
            ))
            
            # Настройка макета
            fig.update_layout(
                title='Корреляционная матрица выбранных признаков',
                xaxis_title='Признаки',
                yaxis_title='Признаки',
                width=max(600, len(available_features) * 40),
                height=max(600, len(available_features) * 40),
                template='plotly_white'
            )
            
            # Поворот подписей осей
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(tickangle=0)
            
            if save_path:
                fig.write_html(save_path)
                print(f"📊 Корреляционная матрица сохранена: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Ошибка создания корреляционной матрицы: {e}")
            return None
    
    def visualize_feature_categories(self, indicators: Dict[str, Any], save_path: str = None) -> go.Figure:
        """
        Визуализация распределения признаков по категориям.
        
        Args:
            indicators: Результаты селекции
            save_path: Путь для сохранения
            
        Returns:
            Plotly Figure объект
        """
        if 'selected_features' not in indicators:
            print("⚠️ Список выбранных признаков недоступен")
            return None
        
        selected_features = indicators['selected_features']
        
        # Категоризация признаков
        categories = {
            'Moving Averages': [],
            'Momentum': [],
            'Volatility': [],
            'Volume': [],
            'Price Features': [],
            'Other': []
        }
        
        for feature in selected_features:
            if any(ma in feature for ma in ['sma_', 'ema_', 'tema', 'trima']):
                categories['Moving Averages'].append(feature)
            elif any(mom in feature for mom in ['rsi_', 'macd', 'stoch', 'williams', 'roc_', 'momentum']):
                categories['Momentum'].append(feature)
            elif any(vol in feature for vol in ['atr_', 'bb_', 'volatility', 'natr']):
                categories['Volatility'].append(feature)
            elif any(vol in feature for vol in ['volume', 'obv', 'ad', 'mfi', 'vwap']):
                categories['Volume'].append(feature)
            elif any(price in feature for price in ['price_change', 'return', 'ratio']):
                categories['Price Features'].append(feature)
            else:
                categories['Other'].append(feature)
        
        # Подготовка данных для графика
        category_names = list(categories.keys())
        category_counts = [len(categories[cat]) for cat in category_names]
        
        # Создание круговой диаграммы
        fig = go.Figure(data=[go.Pie(
            labels=category_names,
            values=category_counts,
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value} (%{percent})',
            hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Доля: %{percent}<extra></extra>',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#000000', width=2)
            )
        )])
        
        # Настройка макета
        fig.update_layout(
            title=f'Распределение выбранных признаков по категориям<br><sub>Всего признаков: {len(selected_features)}</sub>',
            template='plotly_white',
            width=700,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 График категорий сохранен: {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, indicators: Dict[str, Any], data: pd.DataFrame, 
                                  save_dir: str = "feature_selection_report") -> Dict[str, go.Figure]:
        """
        Создание комплексного отчета по селекции признаков.
        
        Args:
            indicators: Результаты селекции
            data: Исходные данные
            save_dir: Директория для сохранения отчетов
            
        Returns:
            Словарь с созданными графиками
        """
        import os
        
        # Создание директории
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📊 Создание комплексного отчета по селекции признаков...")
        
        reports = {}
        
        # 1. График важности признаков
        importance_fig = self.visualize_feature_importance(
            indicators, 
            os.path.join(save_dir, "feature_importance.html")
        )
        if importance_fig:
            reports['importance'] = importance_fig
        
        # 2. Корреляционная матрица
        if 'selected_features' in indicators:
            corr_fig = self.visualize_correlation_matrix(
                data, 
                indicators['selected_features'],
                os.path.join(save_dir, "correlation_matrix.html")
            )
            if corr_fig:
                reports['correlation'] = corr_fig
        
        # 3. Распределение по категориям
        categories_fig = self.visualize_feature_categories(
            indicators,
            os.path.join(save_dir, "feature_categories.html")
        )
        if categories_fig:
            reports['categories'] = categories_fig
        
        # 4. Сводный dashboard
        self._create_dashboard(reports, indicators, os.path.join(save_dir, "dashboard.html"))
        
        print(f"✅ Комплексный отчет создан в директории: {save_dir}")
        return reports
    
    def _create_dashboard(self, reports: Dict[str, go.Figure], indicators: Dict[str, Any], save_path: str):
        """Создание сводного dashboard."""
        try:
            from plotly.subplots import make_subplots
            
            # Создание подграфиков
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Важность признаков', 'Распределение по категориям', 
                               'Статистика селекции', 'Информация о методе'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "table"}, {"type": "table"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 1. Важность признаков (упрощенная версия)
            if 'importance' in reports and 'feature_importance' in indicators:
                feature_importance = indicators['feature_importance']
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                features = [feat[0] for feat in sorted_features]
                importances = [feat[1] for feat in sorted_features]
                
                fig.add_trace(
                    go.Bar(y=features, x=importances, orientation='h', name='Важность'),
                    row=1, col=1
                )
            
            # 2. Категории (упрощенная версия)
            if 'categories' in reports:
                fig.add_trace(reports['categories'].data[0], row=1, col=2)
            
            # 3. Статистика селекции
            stats_data = [
                ["Метод селекции", indicators.get('selection_method', 'N/A')],
                ["Всего признаков", indicators.get('n_features', 'N/A')],
                ["Лучший признак", list(indicators.get('feature_importance', {}).keys())[0] if indicators.get('feature_importance') else 'N/A'],
                ["Средняя важность", f"{np.mean(list(indicators.get('feature_importance', {}).values())):.4f}" if indicators.get('feature_importance') else 'N/A']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Параметр', 'Значение'], fill_color='lightblue'),
                    cells=dict(values=list(zip(*stats_data)), fill_color='white')
                ),
                row=2, col=1
            )
            
            # 4. Информация о процессе
            process_info = [
                ["Этап", "Статус"],
                ["Генерация индикаторов", "✅ Завершено"],
                ["Фильтрация по вариации", "✅ Завершено"],
                ["Корреляционный анализ", "✅ Завершено"],
                ["ML селекция", "✅ Завершено"],
                ["Кросс-валидация", "✅ Завершено"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Этап обработки', 'Статус'], fill_color='lightgreen'),
                    cells=dict(values=list(zip(*process_info)), fill_color='white')
                ),
                row=2, col=2
            )
            
            # Настройка макета
            fig.update_layout(
                title=f'Dashboard анализа селекции признаков<br><sub>{indicators.get("selection_method", "unknown")} метод</sub>',
                height=800,
                showlegend=False,
                template='plotly_white'
            )
            
            # Сохранение
            fig.write_html(save_path)
            print(f"📊 Dashboard сохранен: {save_path}")
            
        except Exception as e:
            print(f"❌ Ошибка создания dashboard: {e}")
    
    def show_interactive_plots(self, indicators: Dict[str, Any], data: pd.DataFrame):
        """Отображение интерактивных графиков в браузере."""
        print("🚀 Запуск интерактивной визуализации...")
        
        # Важность признаков
        importance_fig = self.visualize_feature_importance(indicators)
        if importance_fig:
            importance_fig.show()
        
        # Корреляционная матрица
        if 'selected_features' in indicators:
            corr_fig = self.visualize_correlation_matrix(data, indicators['selected_features'])
            if corr_fig:
                corr_fig.show()
        
        # Категории признаков
        categories_fig = self.visualize_feature_categories(indicators)
        if categories_fig:
            categories_fig.show()
        
        print("✅ Интерактивные графики отображены в браузере")
    
    def get_optimized_config(self, original_config, selected_indicators: Dict[str, Any]) -> Any:
        """Создать оптимизированную конфигурацию с выбранными индикаторами."""
        
        # Создаем новую конфигурацию на основе оригинальной
        optimized_config = original_config
        
        # Анализируем какие типы индикаторов были выбраны
        selected_features = selected_indicators.get('selected_features', [])
        
        # Строим новый indicator_periods на основе выбранных признаков
        new_indicator_periods = {}
        
        for feature in selected_features:
            if 'sma_' in feature:
                period = int(feature.split('_')[1])
                if 'sma' not in new_indicator_periods:
                    new_indicator_periods['sma'] = []
                new_indicator_periods['sma'].append(period)
            
            elif 'ema_' in feature:
                period = int(feature.split('_')[1])
                if 'ema' not in new_indicator_periods:
                    new_indicator_periods['ema'] = []
                new_indicator_periods['ema'].append(period)
            
            elif 'rsi_' in feature:
                period = int(feature.split('_')[1])
                if 'rsi' not in new_indicator_periods:
                    new_indicator_periods['rsi'] = []
                new_indicator_periods['rsi'].append(period)
            
            elif 'macd' in feature:
                if 'macd' not in new_indicator_periods:
                    new_indicator_periods['macd'] = [12, 26, 9]
            
            elif 'bb_' in feature:
                try:
                    period = int(feature.split('_')[2])
                    if 'bollinger' not in new_indicator_periods:
                        new_indicator_periods['bollinger'] = []
                    if period not in new_indicator_periods['bollinger']:
                        new_indicator_periods['bollinger'].append(period)
                except:
                    pass
            
            elif 'atr_' in feature:
                period = int(feature.split('_')[1])
                if 'atr' not in new_indicator_periods:
                    new_indicator_periods['atr'] = []
                new_indicator_periods['atr'].append(period)
            
            elif 'obv' in feature:
                if 'obv' not in new_indicator_periods:
                    new_indicator_periods['obv'] = []
            
            elif 'stoch' in feature:
                if 'stochastic' not in new_indicator_periods:
                    new_indicator_periods['stochastic'] = [14, 3, 3]
        
        # Убираем дубликаты и сортируем
        for key in new_indicator_periods:
            if isinstance(new_indicator_periods[key], list):
                new_indicator_periods[key] = sorted(list(set(new_indicator_periods[key])))
        
        # Обновляем конфигурацию
        optimized_config.indicator_periods = new_indicator_periods
        
        print(f"\n🔧 ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ ИНДИКАТОРОВ:")
        for indicator_type, periods in new_indicator_periods.items():
            print(f"   {indicator_type}: {periods}")
        
        return optimized_config


def create_auto_optimized_config(original_config, data: pd.DataFrame):
    """
    Создать автоматически оптимизированную конфигурацию индикаторов.
    
    Args:
        original_config: Оригинальная конфигурация
        data: Данные для анализа
        
    Returns:
        Оптимизированная конфигурация
    """
    
    selector = AutomaticFeatureSelector(original_config)
    
    # Выбираем лучшие индикаторы
    selected_indicators = selector.select_best_indicators(data)
    
    # Создание интерактивной визуализации
    print("🎨 Создание интерактивной визуализации...")
    selector.show_interactive_plots(selected_indicators, data)
    
    # Создание комплексного отчета
    print("📊 Создание комплексного отчета...")
    reports = selector.create_comprehensive_report(selected_indicators, data)
    
    # Создаем оптимизированную конфигурацию
    optimized_config = selector.get_optimized_config(original_config, selected_indicators)
    
    return optimized_config, selected_indicators


if __name__ == "__main__":
    # Тест автоматической селекции
    from CryptoTrade.ai.STAS_ML.config.ml_config import MLConfig
    
    config = MLConfig(symbol='BTCUSDT')
    
    # Загружаем тестовые данные
    import pandas as pd
    data = pd.read_csv(config.data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    
    # Выполняем автоматическую селекцию
    optimized_config, selected_indicators = create_auto_optimized_config(config, data)
    
    print("✅ Автоматическая селекция завершена!")