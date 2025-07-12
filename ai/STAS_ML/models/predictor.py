"""
Основные ML модели для предсказания криптовалютных цен.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class LSTMModel(nn.Module):
    """LSTM модель для предсказания временных рядов."""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Берем последний выход
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out)
        return predictions


class CryptoPricePredictor:
    """Основной класс для ML предсказания криптовалютных цен."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_type = config.model_type
        self.is_classification = config.target_type == 'direction'
        
    def _create_model(self):
        """Создать модель в зависимости от типа."""
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost не установлен. Установите: pip install xgboost")
            
            if self.is_classification:
                self.model = xgb.XGBClassifier(**self.config.xgb_params)
            else:
                self.model = xgb.XGBRegressor(**self.config.xgb_params)
                
        elif self.model_type == 'random_forest':
            if self.is_classification:
                self.model = RandomForestClassifier(**self.config.rf_params)
            else:
                self.model = RandomForestRegressor(**self.config.rf_params)
                
        elif self.model_type == 'linear':
            if self.is_classification:
                self.model = LogisticRegression(random_state=self.config.random_state, max_iter=1000)
            else:
                self.model = LinearRegression()
                
        elif self.model_type == 'lstm':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch не установлен. Установите: pip install torch")
            # LSTM модель создается отдельно в методе train
            pass
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Обучить модель."""
        print(f"🚀 Обучаем {self.model_type} модель...")
        
        if self.model_type == 'lstm':
            return self._train_lstm(X_train, y_train, X_val, y_val)
        else:
            return self._train_sklearn_model(X_train, y_train, X_val, y_val)
    
    def _train_sklearn_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Обучить sklearn/xgboost модель."""
        self._create_model()
        
        # Обучаем модель
        self.model.fit(X_train, y_train)
        
        # Делаем предсказания
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Вычисляем метрики
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)
        
        print(f"✅ Модель {self.model_type} обучена!")
        self._print_metrics(metrics)
        
        return metrics
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Обучить LSTM модель."""
        # Переформатируем данные для LSTM
        # Предполагаем что X_train имеет shape (samples, features)
        # Преобразуем в (samples, sequence_length, features_per_timestep)
        
        sequence_length = self.config.lookback_window
        features_per_timestep = X_train.shape[1] // sequence_length
        
        X_train_lstm = X_train.reshape(X_train.shape[0], sequence_length, features_per_timestep)
        X_val_lstm = X_val.reshape(X_val.shape[0], sequence_length, features_per_timestep)
        
        # Создаем модель
        input_size = features_per_timestep
        output_size = 1
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.lstm_params['hidden_size'],
            num_layers=self.config.lstm_params['num_layers'],
            output_size=output_size,
            dropout=self.config.lstm_params['dropout']
        )
        
        # Настройка обучения
        criterion = nn.MSELoss() if not self.is_classification else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lstm_params['learning_rate'])
        
        # Создаем DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_lstm), 
            torch.FloatTensor(y_train.reshape(-1, 1))
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.lstm_params['batch_size'], shuffle=True)
        
        # Обучение
        self.model.train()
        train_losses = []
        
        for epoch in range(self.config.lstm_params['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.config.lstm_params['epochs']}, Loss: {avg_loss:.6f}")
        
        # Валидация
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(torch.FloatTensor(X_train_lstm)).numpy().flatten()
            val_pred = self.model(torch.FloatTensor(X_val_lstm)).numpy().flatten()
            
            if self.is_classification:
                train_pred = (torch.sigmoid(torch.FloatTensor(train_pred)) > 0.5).numpy().astype(int)
                val_pred = (torch.sigmoid(torch.FloatTensor(val_pred)) > 0.5).numpy().astype(int)
        
        # Вычисляем метрики
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)
        metrics['train_losses'] = train_losses
        
        print(f"✅ LSTM модель обучена!")
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_train: np.ndarray, train_pred: np.ndarray,
                          y_val: np.ndarray, val_pred: np.ndarray) -> Dict[str, Any]:
        """Вычислить метрики модели."""
        metrics = {}
        
        if self.is_classification:
            # Классификационные метрики
            metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            
            # Дополнительные метрики
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                metrics['val_precision'] = precision_score(y_val, val_pred, average='weighted')
                metrics['val_recall'] = recall_score(y_val, val_pred, average='weighted')
                metrics['val_f1'] = f1_score(y_val, val_pred, average='weighted')
            except:
                pass
                
        else:
            # Регрессионные метрики
            metrics['train_mse'] = mean_squared_error(y_train, train_pred)
            metrics['val_mse'] = mean_squared_error(y_val, val_pred)
            metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
            metrics['val_mae'] = mean_absolute_error(y_val, val_pred)
            
            # R² score
            try:
                from sklearn.metrics import r2_score
                metrics['train_r2'] = r2_score(y_train, train_pred)
                metrics['val_r2'] = r2_score(y_val, val_pred)
            except:
                pass
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Вывести метрики."""
        print("\n📊 Метрики модели:")
        print("-" * 40)
        
        if self.is_classification:
            print(f"Train Accuracy: {metrics.get('train_accuracy', 0):.4f}")
            print(f"Val Accuracy:   {metrics.get('val_accuracy', 0):.4f}")
            if 'val_f1' in metrics:
                print(f"Val F1-score:   {metrics['val_f1']:.4f}")
        else:
            print(f"Train MSE: {metrics.get('train_mse', 0):.6f}")
            print(f"Val MSE:   {metrics.get('val_mse', 0):.6f}")
            print(f"Train MAE: {metrics.get('train_mae', 0):.6f}")
            print(f"Val MAE:   {metrics.get('val_mae', 0):.6f}")
            if 'val_r2' in metrics:
                print(f"Val R²:    {metrics['val_r2']:.4f}")
        
        print("-" * 40)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Сделать предсказания."""
        if self.model is None:
            raise ValueError("Модель не обучена! Вызовите сначала train()")
        
        if self.model_type == 'lstm':
            # Для LSTM нужно переформатировать данные
            sequence_length = self.config.lookback_window
            features_per_timestep = X.shape[1] // sequence_length
            X_lstm = X.reshape(X.shape[0], sequence_length, features_per_timestep)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.FloatTensor(X_lstm)).numpy().flatten()
                
                if self.is_classification:
                    predictions = (torch.sigmoid(torch.FloatTensor(predictions)) > 0.5).numpy().astype(int)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Получить вероятности для классификации."""
        if not self.is_classification:
            raise ValueError("predict_proba доступен только для классификации")
        
        if self.model_type == 'lstm':
            sequence_length = self.config.lookback_window
            features_per_timestep = X.shape[1] // sequence_length
            X_lstm = X.reshape(X.shape[0], sequence_length, features_per_timestep)
            
            self.model.eval()
            with torch.no_grad():
                logits = self.model(torch.FloatTensor(X_lstm)).numpy().flatten()
                probabilities = torch.sigmoid(torch.FloatTensor(logits)).numpy()
                
            # Возвращаем вероятности для обеих классов
            return np.column_stack([1 - probabilities, probabilities])
        else:
            return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Сохранить модель."""
        if self.model is None:
            raise ValueError("Нет модели для сохранения")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'config': self.config,
            'is_classification': self.is_classification
        }
        
        if self.model_type == 'lstm':
            # Для PyTorch моделей сохраняем state_dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_size': self.model.lstm.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'output_size': 1,
                    'dropout': self.config.lstm_params['dropout']
                },
                'config': self.config,
                'model_type': self.model_type,
                'is_classification': self.is_classification
            }, filepath)
        else:
            joblib.dump(model_data, filepath)
        
        print(f"✅ Модель сохранена: {filepath}")
    
    def load(self, filepath: str):
        """Загрузить модель."""
        if self.model_type == 'lstm':
            checkpoint = torch.load(filepath)
            model_config = checkpoint['model_config']
            
            self.model = LSTMModel(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.config = checkpoint['config']
            self.is_classification = checkpoint['is_classification']
        else:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.config = model_data['config']
            self.is_classification = model_data['is_classification']
        
        print(f"✅ Модель загружена: {filepath}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Получить важность признаков (для tree-based моделей)."""
        if self.model_type in ['xgboost', 'random_forest'] and hasattr(self.model, 'feature_importances_'):
            return dict(zip(range(len(self.model.feature_importances_)), self.model.feature_importances_))
        else:
            print("Важность признаков недоступна для данного типа модели")
            return None