from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                     # Main page
    path('about/', views.about, name='about'),               # About page
    path('predict_ajax/', views.predict_ajax, name='predict_ajax'),  # AJAX endpoint
]
