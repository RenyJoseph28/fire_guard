from django.db import models
from django.contrib.auth.models import User

class Alert(models.Model):
    SEVERITY_CHOICES = [
        ('low', 'Low (Smoke/Fog)'),
        ('medium', 'Medium (Burning Material)'),
        ('high', 'High (Fire)'),
        ('critical', 'CRITICAL (Class B / Explosion Risk)'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=100, default="Camera 1")
    alert_type = models.CharField(max_length=100)  # e.g., "Fire", "Class B Fire - Flammable Liquids"
    confidence = models.FloatField()
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='medium')
    is_resolved = models.BooleanField(default=False)
    snapshot = models.ImageField(upload_to='alerts/', null=True, blank=True)

    def __str__(self):
        return f"{self.alert_type} detected at {self.timestamp}"

class AlertRecipient(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    receive_critical_only = models.BooleanField(default=False, help_text="If checked, only High/Critical alerts will be sent.")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.email})"

class FCMDevice(models.Model):
    """Stores Firebase Cloud Messaging tokens for push notifications"""
    token = models.CharField(max_length=500, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    device_name = models.CharField(max_length=100, blank=True, default="Unknown Device")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"FCM Device: {self.device_name} ({self.token[:20]}...)"
