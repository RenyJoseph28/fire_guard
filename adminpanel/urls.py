from django.urls import path
from . import views

app_name = 'adminpanel'

urlpatterns = [
    path('', views.admin_dashboard, name='admin_dashboard'),
    path('login/', views.admin_login, name='admin_login'),
    path('upload/', views.upload_video, name='upload_video'),
    path('recipients/', views.recipients_list, name='recipients_list'),
    path('recipients/add/', views.add_recipient, name='add_recipient'),
    path('recipients/delete/<int:recipient_id>/', views.delete_recipient, name='delete_recipient'),
    
    # API Endpoints
    path('api/save-fcm-token/', views.save_fcm_token, name='save_fcm_token'),
]
