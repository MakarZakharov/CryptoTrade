from django.urls import path
from . import views

urlpatterns = [
    path('', views.comparison_view, name='comparison'),
    path('api/load-parquet/', views.load_parquet_data, name='load_parquet'),
    path('api/available-files/', views.get_available_files, name='available_files'),
]