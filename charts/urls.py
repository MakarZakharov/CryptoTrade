from django.urls import path
from . import views

urlpatterns = [
    path('comparison/', views.comparison_view, name='comparison'),
]