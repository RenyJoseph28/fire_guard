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
