from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Alert
from django.core.files.storage import FileSystemStorage
from .detect_utils import FireDetector
import os

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Hardcoded admin credentials check
        if username == 'admin' and password == 'admin':
            user = User.objects.filter(username='admin').first()
            if not user:
                # Create the admin user if it doesn't exist
                user = User.objects.create_superuser('admin', 'admin@example.com', 'admin')
            
            # Force login with the default backend
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            return redirect('adminpanel:admin_dashboard')

        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_superuser:
                login(request, user)
                return redirect('adminpanel:admin_dashboard')
            else:
                messages.error(request, "Access denied. Admin privileges required.")
        else:
            messages.error(request, "Invalid username or password.")
            
    return render(request, 'adminpanel/login.html')

def admin_dashboard(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')
    
    # Get recent alerts from database
    recent_alerts = Alert.objects.order_by('-timestamp')[:10]
    
    context = {
        'recent_alerts': recent_alerts,
        'alert_count': Alert.objects.count(),
        'high_severity_count': Alert.objects.filter(severity='high').count()
    }
    return render(request, 'adminpanel/dashboard.html', context)

def upload_video(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')

    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        file_path = fs.path(filename)

        # Process the video
        try:
            alerts_count = FireDetector.process_video(file_path)
            messages.success(request, f"Video processed successfully. {alerts_count} potential threats detected.")
        except Exception as e:
            messages.error(request, f"Error processing video: {str(e)}")
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

        return redirect('adminpanel:admin_dashboard')

    return render(request, 'adminpanel/upload_video.html')
