"""Логирование для DRL системы."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


class DRLLogger:
    """Централизованная система логирования для DRL."""
    
    def __init__(self, name: str, log_level: str = "INFO", log_dir: Optional[str] = None):
        """
        Инициализация логгера.
        
        Args:
            name: имя логгера
            log_level: уровень логирования (DEBUG, INFO, WARNING, ERROR)
            log_dir: директория для сохранения логов
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Создаем директорию для логов
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Настраиваем форматирование
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый хендлер с поддержкой UTF-8 и ротацией
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'  # Поддержка UTF-8 для эмодзи
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Настройка консольного хендлера для Windows совместимости
        self._setup_console_compatibility()
        
        self.info(f"Логгер {name} инициализирован. Лог файл: {log_file}")
    
    def _setup_console_compatibility(self):
        """Настройка совместимости консоли для Windows."""
        # Определяем Windows консоль
        self.is_windows_console = (
            sys.platform.startswith('win') and
            hasattr(sys.stdout, 'encoding') and
            sys.stdout.encoding in ['cp1251', 'cp866', 'windows-1251']
        )
    
    def _safe_message_for_console(self, message: str) -> str:
        """Безопасная обработка сообщений для Windows консоли."""
        if not self.is_windows_console:
            return message
        
        # Замена эмодзи на ASCII символы для Windows консоли
        emoji_replacements = {
            '🎉': '[SUCCESS]',
            '💾': '[SAVE]', 
            '📊': '[STATS]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '✅': '[OK]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '💰': '[MONEY]',
            '🔥': '[HOT]',
            '🚀': '[ROCKET]',
            '⭐': '[STAR]',
            '🎯': '[TARGET]',
            '🔔': '[BELL]',
            '📢': '[ANNOUNCE]'
        }
        
        safe_message = message
        for emoji, replacement in emoji_replacements.items():
            safe_message = safe_message.replace(emoji, replacement)
        
        return safe_message
    
    def debug(self, message: str):
        """Логирование отладочной информации."""
        safe_message = self._safe_message_for_console(message)
        self.logger.debug(safe_message)
    
    def info(self, message: str):
        """Логирование информационных сообщений."""
        safe_message = self._safe_message_for_console(message)
        self.logger.info(safe_message)
    
    def warning(self, message: str):
        """Логирование предупреждений."""
        safe_message = self._safe_message_for_console(message)
        self.logger.warning(safe_message)
    
    def error(self, message: str):
        """Логирование ошибок."""
        safe_message = self._safe_message_for_console(message)
        self.logger.error(safe_message)
    
    def critical(self, message: str):
        """Логирование критических ошибок."""
        safe_message = self._safe_message_for_console(message)
        self.logger.critical(safe_message)