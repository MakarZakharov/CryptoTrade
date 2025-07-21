"""Вспомогательные функции для DRL системы."""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Сохранение данных в JSON файл."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Загрузка данных из JSON файла."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """Сохранение данных в pickle файл."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Загрузка данных из pickle файла."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def create_experiment_name(symbol: str, agent_type: str, timestamp: Optional[str] = None) -> str:
    """Создание имени эксперимента."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{agent_type}_{timestamp}"


def normalize_data(data: Union[np.ndarray, pd.DataFrame], method: str = "minmax") -> Union[np.ndarray, pd.DataFrame]:
    """Нормализация данных."""
    if method == "minmax":
        return (data - data.min()) / (data.max() - data.min())
    elif method == "zscore":
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")


def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Валидация данных."""
    if data.empty:
        return False
    for col in required_columns:
        if col not in data.columns:
            return False
    return True


def split_data(data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1) -> tuple:
    """Разделение данных на train/validation/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Сумма ratios должна быть равна 1.0"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data


def ensure_directory(path: str) -> None:
    """Создание директории если она не существует."""
    os.makedirs(path, exist_ok=True)


def get_file_size(filepath: str) -> int:
    """Получение размера файла в байтах."""
    return os.path.getsize(filepath) if os.path.exists(filepath) else 0


def format_bytes(size: int) -> str:
    """Форматирование размера в человекочитаемый формат."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"