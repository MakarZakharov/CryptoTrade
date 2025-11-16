from django.urls import path
from . import views

urlpatterns = [
    path('', views.comparison_view, name='comparison'),
    path('api/load-parquet/', views.load_parquet_data, name='load_parquet'),
    path('api/get-columns/', views.get_parquet_columns, name='get_columns'),
]