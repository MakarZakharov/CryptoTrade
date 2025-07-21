"""Менеджер экспериментов для DRL."""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from ..utils import DRLLogger


class ExperimentManager:
    """
    Менеджер экспериментов для управления DRL экспериментами.
    
    Отвечает за:
    - Создание и управление директориями экспериментов
    - Сохранение конфигураций и результатов
    - Сравнение экспериментов
    - Архивирование и восстановление
    """
    
    def __init__(
        self,
        base_dir: str = "CryptoTrade/ai/DRL/experiments",
        experiment_name: Optional[str] = None,
        auto_create: bool = True
    ):
        """
        Инициализация менеджера экспериментов.
        
        Args:
            base_dir: базовая директория для экспериментов
            experiment_name: имя эксперимента (если не указано, генерируется автоматически)
            auto_create: автоматически создавать директории
        """
        self.base_dir = Path(base_dir)
        self.logger = DRLLogger("experiment_manager")
        
        # Генерация или установка имени эксперимента
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Пути к директориям эксперимента
        self.experiment_dir = self.base_dir / self.experiment_name
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.results_dir = self.experiment_dir / "results"
        self.configs_dir = self.experiment_dir / "configs"
        self.plots_dir = self.experiment_dir / "plots"
        
        # Создание директорий
        if auto_create:
            self.create_directories()
        
        self.logger.info(f"Менеджер экспериментов инициализирован: {self.experiment_name}")
    
    def create_directories(self):
        """Создание всех необходимых директорий."""
        directories = [
            self.experiment_dir,
            self.models_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.results_dir,
            self.configs_dir,
            self.plots_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"Директории созданы для эксперимента: {self.experiment_name}")
    
    def save_experiment_metadata(self, metadata: Dict[str, Any]):
        """
        Сохранение метаданных эксперимента.
        
        Args:
            metadata: словарь с метаданными
        """
        # Добавляем системную информацию
        system_info = self._get_system_info()
        metadata.update({
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'system_info': system_info
        })
        
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Метаданные эксперимента сохранены: {metadata_path}")
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """
        Сохранение конфигурации.
        
        Args:
            config_name: имя конфигурации
            config_data: данные конфигурации
        """
        config_path = self.configs_dir / f"{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        self.logger.debug(f"Конфигурация сохранена: {config_path}")
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Загрузка конфигурации.
        
        Args:
            config_name: имя конфигурации
            
        Returns:
            Загруженная конфигурация или None
        """
        config_path = self.configs_dir / f"{config_name}.json"
        
        if not config_path.exists():
            self.logger.warning(f"Конфигурация не найдена: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.logger.debug(f"Конфигурация загружена: {config_path}")
            return config_data
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации {config_path}: {e}")
            return None
    
    def save_results(self, results_name: str, results_data: Union[Dict, pd.DataFrame]):
        """
        Сохранение результатов эксперимента.
        
        Args:
            results_name: имя файла результатов
            results_data: данные результатов
        """
        if isinstance(results_data, pd.DataFrame):
            # Сохраняем DataFrame как CSV и JSON
            csv_path = self.results_dir / f"{results_name}.csv"
            results_data.to_csv(csv_path, index=False)
            
            json_path = self.results_dir / f"{results_name}.json"
            results_data.to_json(json_path, orient='records', indent=2)
            
            self.logger.debug(f"Результаты DataFrame сохранены: {csv_path}, {json_path}")
            
        else:
            # Сохраняем как JSON
            json_path = self.results_dir / f"{results_name}.json"
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.debug(f"Результаты сохранены: {json_path}")
    
    def load_results(self, results_name: str, format: str = 'auto') -> Optional[Union[Dict, pd.DataFrame]]:
        """
        Загрузка результатов эксперимента.
        
        Args:
            results_name: имя файла результатов
            format: формат загрузки ('auto', 'json', 'csv')
            
        Returns:
            Загруженные результаты
        """
        if format == 'auto':
            # Пробуем разные форматы
            csv_path = self.results_dir / f"{results_name}.csv"
            json_path = self.results_dir / f"{results_name}.json"
            
            if csv_path.exists():
                format = 'csv'
            elif json_path.exists():
                format = 'json'
            else:
                self.logger.warning(f"Результаты не найдены: {results_name}")
                return None
        
        try:
            if format == 'csv':
                csv_path = self.results_dir / f"{results_name}.csv"
                results = pd.read_csv(csv_path)
                self.logger.debug(f"Результаты CSV загружены: {csv_path}")
                return results
                
            elif format == 'json':
                json_path = self.results_dir / f"{results_name}.json"
                with open(json_path, 'r') as f:
                    results = json.load(f)
                self.logger.debug(f"Результаты JSON загружены: {json_path}")
                return results
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки результатов {results_name}: {e}")
            return None
    
    def list_experiments(self) -> List[str]:
        """
        Получение списка всех экспериментов.
        
        Returns:
            Список имен экспериментов
        """
        if not self.base_dir.exists():
            return []
        
        experiments = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "experiment_metadata.json").exists():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def get_experiment_info(self, experiment_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Получение информации об эксперименте.
        
        Args:
            experiment_name: имя эксперимента (текущий если не указан)
            
        Returns:
            Информация об эксперименте
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment_dir = self.base_dir / experiment_name
        metadata_path = experiment_dir / "experiment_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Добавляем статистику файлов
            file_stats = self._get_experiment_file_stats(experiment_dir)
            metadata['file_stats'] = file_stats
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации об эксперименте {experiment_name}: {e}")
            return None
    
    def _get_experiment_file_stats(self, experiment_dir: Path) -> Dict[str, Any]:
        """Получение статистики файлов эксперимента."""
        stats = {
            'total_size_mb': 0,
            'file_counts': {},
            'directories': []
        }
        
        try:
            for item in experiment_dir.rglob('*'):
                if item.is_file():
                    # Размер файла
                    size_mb = item.stat().st_size / (1024 * 1024)
                    stats['total_size_mb'] += size_mb
                    
                    # Счетчик файлов по типам
                    suffix = item.suffix or 'no_extension'
                    stats['file_counts'][suffix] = stats['file_counts'].get(suffix, 0) + 1
                
                elif item.is_dir() and item != experiment_dir:
                    relative_path = item.relative_to(experiment_dir)
                    stats['directories'].append(str(relative_path))
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
        except Exception as e:
            self.logger.warning(f"Ошибка получения статистики файлов: {e}")
        
        return stats
    
    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """
        Сравнение экспериментов.
        
        Args:
            experiment_names: список имен экспериментов для сравнения
            
        Returns:
            DataFrame с сравнением экспериментов
        """
        comparison_data = []
        
        for exp_name in experiment_names:
            exp_info = self.get_experiment_info(exp_name)
            if exp_info is None:
                continue
            
            # Извлекаем ключевые метрики
            row = {
                'experiment_name': exp_name,
                'created_at': exp_info.get('created_at', ''),
                'agent_type': exp_info.get('agent_type', ''),
                'symbol': exp_info.get('symbol', ''),
                'total_timesteps': exp_info.get('total_timesteps', 0),
                'total_size_mb': exp_info.get('file_stats', {}).get('total_size_mb', 0)
            }
            
            # Добавляем метрики производительности если есть
            if 'final_performance' in exp_info:
                perf = exp_info['final_performance']
                row.update({
                    'mean_reward': perf.get('mean_reward', 0),
                    'total_return': perf.get('total_return', 0),
                    'mean_portfolio_value': perf.get('mean_portfolio_value', 0)
                })
            
            comparison_data.append(row)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Сохраняем сравнение
        comparison_path = self.results_dir / "experiments_comparison.csv"
        df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"Сравнение экспериментов сохранено: {comparison_path}")
        
        return df
    
    def archive_experiment(self, experiment_name: Optional[str] = None, archive_dir: Optional[str] = None) -> str:
        """
        Архивирование эксперимента.
        
        Args:
            experiment_name: имя эксперимента (текущий если не указан)
            archive_dir: директория для архивов
            
        Returns:
            Путь к созданному архиву
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        if archive_dir is None:
            archive_dir = self.base_dir / "archives"
        
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_dir = self.base_dir / experiment_name
        if not experiment_dir.exists():
            raise ValueError(f"Эксперимент не найден: {experiment_name}")
        
        # Создаем архив
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{experiment_name}_{timestamp}"
        archive_path = archive_dir / archive_name
        
        # Используем shutil для создания zip архива
        shutil.make_archive(str(archive_path), 'zip', str(experiment_dir))
        
        final_archive_path = str(archive_path) + '.zip'
        
        self.logger.info(f"Эксперимент заархивирован: {final_archive_path}")
        
        return final_archive_path
    
    def delete_experiment(self, experiment_name: str, confirm: bool = False):
        """
        Удаление эксперимента.
        
        Args:
            experiment_name: имя эксперимента
            confirm: подтверждение удаления
        """
        if not confirm:
            raise ValueError("Для удаления эксперимента требуется подтверждение (confirm=True)")
        
        experiment_dir = self.base_dir / experiment_name
        if not experiment_dir.exists():
            self.logger.warning(f"Эксперимент не найден: {experiment_name}")
            return
        
        # Удаляем директорию
        shutil.rmtree(str(experiment_dir))
        
        self.logger.info(f"Эксперимент удален: {experiment_name}")
    
    def get_best_experiments(self, metric: str = 'mean_reward', top_k: int = 5) -> pd.DataFrame:
        """
        Получение лучших экспериментов по метрике.
        
        Args:
            metric: метрика для сортировки
            top_k: количество лучших экспериментов
            
        Returns:
            DataFrame с лучшими экспериментами
        """
        experiments = self.list_experiments()
        if not experiments:
            return pd.DataFrame()
        
        comparison_df = self.compare_experiments(experiments)
        if comparison_df.empty:
            return pd.DataFrame()
        
        # Сортируем по метрике
        if metric in comparison_df.columns:
            sorted_df = comparison_df.sort_values(metric, ascending=False)
            return sorted_df.head(top_k)
        else:
            self.logger.warning(f"Метрика {metric} не найдена в данных экспериментов")
            return comparison_df.head(top_k)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе."""
        import platform
        import psutil
        
        try:
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'hostname': platform.node()
            }
            
            # Информация о GPU если доступна
            try:
                import torch
                if torch.cuda.is_available():
                    system_info['gpu_available'] = True
                    system_info['gpu_name'] = torch.cuda.get_device_name(0)
                    system_info['gpu_memory_gb'] = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    )
                else:
                    system_info['gpu_available'] = False
            except ImportError:
                system_info['gpu_available'] = False
            
        except Exception as e:
            self.logger.warning(f"Ошибка получения системной информации: {e}")
            system_info = {'error': str(e)}
        
        return system_info
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Генерация сводки по всем экспериментам."""
        experiments = self.list_experiments()
        
        summary = {
            'total_experiments': len(experiments),
            'experiments_by_agent': {},
            'experiments_by_symbol': {},
            'total_disk_usage_mb': 0,
            'recent_experiments': []
        }
        
        for exp_name in experiments:
            exp_info = self.get_experiment_info(exp_name)
            if exp_info is None:
                continue
            
            # Статистика по агентам
            agent_type = exp_info.get('agent_type', 'unknown')
            summary['experiments_by_agent'][agent_type] = summary['experiments_by_agent'].get(agent_type, 0) + 1
            
            # Статистика по символам
            symbol = exp_info.get('symbol', 'unknown')
            summary['experiments_by_symbol'][symbol] = summary['experiments_by_symbol'].get(symbol, 0) + 1
            
            # Размер на диске
            disk_usage = exp_info.get('file_stats', {}).get('total_size_mb', 0)
            summary['total_disk_usage_mb'] += disk_usage
            
            # Недавние эксперименты
            if len(summary['recent_experiments']) < 5:
                summary['recent_experiments'].append({
                    'name': exp_name,
                    'created_at': exp_info.get('created_at', ''),
                    'agent_type': agent_type,
                    'symbol': symbol
                })
        
        # Округляем размер диска
        summary['total_disk_usage_mb'] = round(summary['total_disk_usage_mb'], 2)
        
        # Сохраняем сводку
        summary_path = self.base_dir / "experiments_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Сводка экспериментов сохранена: {summary_path}")
        
        return summary
    
    def cleanup_old_experiments(
        self, 
        keep_recent: int = 10,
        keep_best: int = 5, 
        metric: str = 'mean_reward',
        dry_run: bool = True
    ) -> List[str]:
        """
        Очистка старых экспериментов.
        
        Args:
            keep_recent: количество недавних экспериментов для сохранения
            keep_best: количество лучших экспериментов для сохранения
            metric: метрика для определения лучших
            dry_run: только показать что будет удалено, не удалять
            
        Returns:
            Список экспериментов для удаления
        """
        experiments = self.list_experiments()
        if len(experiments) <= keep_recent:
            return []
        
        # Получаем информацию о всех экспериментах
        exp_info_list = []
        for exp_name in experiments:
            exp_info = self.get_experiment_info(exp_name)
            if exp_info:
                exp_info['name'] = exp_name
                exp_info_list.append(exp_info)
        
        # Сортируем по дате создания (недавние)
        exp_info_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        recent_experiments = set(exp['name'] for exp in exp_info_list[:keep_recent])
        
        # Сортируем по метрике (лучшие)
        best_experiments = set()
        if metric in ['mean_reward', 'total_return']:  # метрики, которые могут быть в данных
            metric_experiments = [exp for exp in exp_info_list if metric in exp.get('final_performance', {})]
            if metric_experiments:
                metric_experiments.sort(
                    key=lambda x: x.get('final_performance', {}).get(metric, float('-inf')), 
                    reverse=True
                )
                best_experiments = set(exp['name'] for exp in metric_experiments[:keep_best])
        
        # Определяем эксперименты для сохранения
        keep_experiments = recent_experiments | best_experiments
        
        # Определяем эксперименты для удаления
        to_delete = [exp for exp in experiments if exp not in keep_experiments]
        
        if dry_run:
            self.logger.info(f"DRY RUN: Будет удалено {len(to_delete)} экспериментов:")
            for exp in to_delete:
                self.logger.info(f"  - {exp}")
        else:
            for exp in to_delete:
                try:
                    self.delete_experiment(exp, confirm=True)
                except Exception as e:
                    self.logger.error(f"Ошибка удаления эксперимента {exp}: {e}")
            
            self.logger.info(f"Удалено {len(to_delete)} экспериментов")
        
        return to_delete