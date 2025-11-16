from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import os


def comparison_view(request):
    """Головна сторінка з графіками"""
    return render(request, 'charts/comparison.html')


def load_parquet_data(request):
    """API endpoint для завантаження даних з Parquet файлів"""
    if request.method == 'POST':
        try:
            file1_path = request.POST.get('file1_path')
            file2_path = request.POST.get('file2_path')
            file1_upload = request.FILES.get('file1')
            file2_upload = request.FILES.get('file2')

            # Обробляємо перший файл
            if file1_upload:
                df1 = pd.read_parquet(file1_upload)
            elif file1_path and os.path.exists(file1_path):
                df1 = pd.read_parquet(file1_path)
            else:
                df1 = None

            # Обробляємо другий файл
            if file2_upload:
                df2 = pd.read_parquet(file2_upload)
            elif file2_path and os.path.exists(file2_path):
                df2 = pd.read_parquet(file2_path)
            else:
                df2 = None

            # Конвертуємо DataFrame в дані для графіків
            data1 = process_dataframe(df1) if df1 is not None else []
            data2 = process_dataframe(df2) if df2 is not None else []

            return JsonResponse({
                'success': True,
                'data1': data1,
                'data2': data2,
                'columns1': list(df1.columns) if df1 is not None else [],
                'columns2': list(df2.columns) if df2 is not None else [],
                'info1': get_dataframe_info(df1) if df1 is not None else {},
                'info2': get_dataframe_info(df2) if df2 is not None else {},
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)


def process_dataframe(df, max_points=1000):
    """Обробляє DataFrame та повертає дані для графіка"""
    if df is None or df.empty:
        return []

    # Знаходимо числові колонки
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        return []

    # Якщо є datetime індекс, використовуємо його як X
    if pd.api.types.is_datetime64_any_dtype(df.index):
        x_data = (df.index - df.index[0]).total_seconds()
        y_col = numeric_cols[0]
    # Якщо є 2+ числові колонки
    elif len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_data = df[x_col]
    # Якщо тільки 1 числова колонка
    else:
        x_data = df.index
        y_col = numeric_cols[0]

    # Створюємо дані
    data = []
    step = max(1, len(df) // max_points)

    for i in range(0, len(df), step):
        try:
            data.append({
                'x': float(x_data.iloc[i] if hasattr(x_data, 'iloc') else x_data[i]),
                'y': float(df[y_col].iloc[i])
            })
        except (ValueError, TypeError):
            continue

    return data


def get_dataframe_info(df):
    """Повертає інформацію про DataFrame"""
    if df is None:
        return {}

    return {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'has_datetime_index': pd.api.types.is_datetime64_any_dtype(df.index)
    }


def get_available_files(request):
    """API для отримання списку доступних Parquet файлів"""
    # Шукаємо у папці Date
    data_dir = os.path.join(settings.BASE_DIR, 'Date')

    files = []

    # Рекурсивно шукаємо всі parquet файли
    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.parquet'):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, data_dir)

                    # Перевіряємо розмір
                    file_size = os.path.getsize(filepath)
                    if file_size == 0:
                        print(f"Skipping empty file: {filepath}")
                        continue

                    try:
                        df = pd.read_parquet(filepath)
                        if len(df) > 0:
                            files.append({
                                'name': filename,
                                'path': rel_path,
                                'full_path': filepath,
                                'size': file_size,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'numeric_columns': list(df.select_dtypes(include=['number']).columns)
                            })
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
                        pass
    else:
        print(f"Directory not found: {data_dir}")

    return JsonResponse({
        'success': True,
        'files': files
    })