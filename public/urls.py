from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='public_home'),
    path('signin/', views.signin, name='public_signin'),
    path('signup/', views.signup, name='public_signup'),
]
