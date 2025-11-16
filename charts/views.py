from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json


def comparison_view(request):
    return render(request, 'charts/comparison.html')


def load_parquet_data(request):
    """API endpoint для завантаження даних з Parquet файлів"""
    if request.method == 'POST' and request.FILES:
        try:
            file1 = request.FILES.get('file1')
            file2 = request.FILES.get('file2')

            # Читаємо перший файл
            if file1:
                df1 = pd.read_parquet(file1)
                # Беремо перші 2 числові колонки або вказані
                numeric_cols1 = df1.select_dtypes(include=['number']).columns[:2]
                data1 = []
                for idx, row in df1.iterrows():
                    if len(numeric_cols1) >= 2:
                        data1.append({
                            'x': float(row[numeric_cols1[0]]),
                            'y': float(row[numeric_cols1[1]])
                        })
                    elif len(numeric_cols1) == 1:
                        data1.append({
                            'x': idx,
                            'y': float(row[numeric_cols1[0]])
                        })
            else:
                data1 = []

            # Читаємо другий файл
            if file2:
                df2 = pd.read_parquet(file2)
                numeric_cols2 = df2.select_dtypes(include=['number']).columns[:2]
                data2 = []
                for idx, row in df2.iterrows():
                    if len(numeric_cols2) >= 2:
                        data2.append({
                            'x': float(row[numeric_cols2[0]]),
                            'y': float(row[numeric_cols2[1]])
                        })
                    elif len(numeric_cols2) == 1:
                        data2.append({
                            'x': idx,
                            'y': float(row[numeric_cols2[0]])
                        })
            else:
                data2 = []

            return JsonResponse({
                'success': True,
                'data1': data1[:1000],  # Обмежуємо до 1000 точок
                'data2': data2[:1000],
                'columns1': list(df1.columns) if file1 else [],
                'columns2': list(df2.columns) if file2 else [],
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'success': False, 'error': 'No files provided'}, status=400)


def get_parquet_columns(request):
    """Отримати список колонок з Parquet файлу"""
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            df = pd.read_parquet(file)

            return JsonResponse({
                'success': True,
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=['number']).columns),
                'shape': df.shape,
                'sample': df.head(5).to_dict('records')
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'success': False, 'error': 'No file provided'}, status=400)