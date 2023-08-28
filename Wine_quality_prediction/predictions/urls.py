# data_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('Home/', views.Home_view, name='Home'),
    path('wine_quality/', views.wine_quality_view, name='wine_quality'),
    path('success/', views.success_view, name='success'),
    #path('data/', views.your_view, name='data'),
    #path('redirect_url/', views.your_view, name='data'),
    path('data/<str:values>/', views.your_view, name='data'),
]
