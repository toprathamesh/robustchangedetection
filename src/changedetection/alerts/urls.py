from django.urls import path
from . import views

app_name = 'alerts'

urlpatterns = [
    path('send-test/', views.send_test_alert, name='send_test_alert'),
    path('configure/<uuid:aoi_id>/', views.configure_alerts, name='configure_alerts'),
] 