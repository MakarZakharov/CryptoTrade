from django.urls import path
from . import views

urlpatterns = [
    path('', views.comparison_view, name='comparison'),
    path('btc/', views.real_data_view, name='real_data'),
    path('api/load-parquet/', views.load_parquet_data, name='load_parquet'),
    path('api/load-btc/', views.load_btc_data, name='load_btc'),
    path('api/load-btc-custom/', views.load_btc_custom_columns, name='load_btc_custom'),
    path('api/get-btc-columns/', views.get_btc_columns, name='get_btc_columns'),
    path('api/available-files/', views.get_available_files, name='available_files'),  # Додайте цей рядок
    path('api/load-custom-columns/', views.load_custom_columns, name='load_custom_columns'),
    path('api/get-columns/', views.get_parquet_columns, name='get_columns'),
]