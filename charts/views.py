from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import os
from datetime import datetime


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
            result1 = process_dataframe_with_dates(df1) if df1 is not None else {'data': [], 'dates': []}
            result2 = process_dataframe_with_dates(df2) if df2 is not None else {'data': [], 'dates': []}

            return JsonResponse({
                'success': True,
                'data1': result1['data'],
                'data2': result2['data'],
                'dates1': result1['dates'],
                'dates2': result2['dates'],
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


def process_dataframe_with_dates(df, max_points=1000):
    """Обробляє DataFrame та повертає дані для графіка з датами"""
    if df is None or df.empty:
        return {'data': [], 'dates': []}

    print(f"Processing dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Знаходимо числові колонки
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        return {'data': [], 'dates': []}

    # Шукаємо колонку з датою/часом
    date_col = None
    for col in ['timestamp', 'date', 'time', 'datetime', 'open_time']:
        if col in df.columns:
            date_col = col
            print(f"Found date column: {date_col}")
            break

    # Визначаємо Y колонку
    if 'close' in numeric_cols:
        y_col = 'close'
    elif len(numeric_cols) >= 2:
        y_col = numeric_cols[1] if numeric_cols[0] in ['timestamp', 'time'] else numeric_cols[0]
    else:
        y_col = numeric_cols[0]

    print(f"Using Y column: {y_col}")

    # Створюємо дані
    data = []
    dates = []
    step = max(1, len(df) // max_points)

    for i in range(0, len(df), step):
        try:
            data.append({
                'x': float(i),
                'y': float(df[y_col].iloc[i])
            })

            # Додаємо дату якщо є
            if date_col:
                date_value = df[date_col].iloc[i]
                # Конвертуємо timestamp в дату
                if pd.api.types.is_numeric_dtype(df[date_col]):
                    # Якщо це мілісекунди
                    if date_value > 1e12:
                        date_value = date_value / 1000
                    date_str = datetime.fromtimestamp(date_value).strftime('%Y-%m-%d')
                else:
                    date_str = str(date_value)[:10]  # Беремо тільки дату
                dates.append(date_str)
            else:
                dates.append(str(i))

        except (ValueError, TypeError) as e:
            print(f"Error at index {i}: {e}")
            continue

    print(f"Generated {len(data)} data points")
    if len(data) > 0:
        print(f"Sample data: {data[0]}, {data[-1]}")
        print(f"Sample dates: {dates[0] if dates else 'N/A'}, {dates[-1] if dates else 'N/A'}")

    return {'data': data, 'dates': dates}


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
    data_dir = os.path.join(settings.BASE_DIR, 'EnvironmentData', 'Date', 'binance')

    print(f"Looking for files in: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")

    files = []

    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.parquet'):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, data_dir)

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

    print(f"Found {len(files)} valid files")

    return JsonResponse({
        'success': True,
        'files': files
    })