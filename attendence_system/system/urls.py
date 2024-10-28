from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('system/', views.system, name='attendence-system'),
    path('recognition/', views.recognition, name='attendence-system'),
    path('students/', views.get_students, name='get_students')
]
