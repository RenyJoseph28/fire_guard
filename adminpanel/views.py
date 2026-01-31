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

    print(f"DEBUG: upload_video view called. Method: {request.method}")
    if request.method == 'POST':
        if request.FILES.get('video_file'):
            try:
                video_file = request.FILES['video_file']
                print(f"DEBUG: Receiving file: {video_file.name}, size: {video_file.size}")
                
                fs = FileSystemStorage()
                filename = fs.save(video_file.name, video_file)
                file_path = fs.path(filename)
                print(f"DEBUG: Saved file to: {file_path}")

                # Process the video
                print("DEBUG: Starting processing...")
                alerts_count = FireDetector.process_video(file_path)
                print(f"DEBUG: Processing complete. Alerts found: {alerts_count}")
                
                messages.success(request, f"Video processed successfully. {alerts_count} potential threats detected.")
            except Exception as e:
                import traceback
                print(f"ERROR in upload_video: {str(e)}")
                traceback.print_exc()
                messages.error(request, f"Error processing video: {str(e)}")
            finally:
                # Clean up uploaded file
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
                    print("DEBUG: Cleaned up uploaded file.")
        else:
            print("DEBUG: No 'video_file' in request.FILES")

        # Redirect with parameter to trigger alarm if alerts were found
        from django.urls import reverse
        from django.http import HttpResponseRedirect
        if 'alerts_count' in locals() and alerts_count > 0:
            return HttpResponseRedirect(reverse('adminpanel:admin_dashboard') + '?uploaded=true')
        return redirect('adminpanel:admin_dashboard')

    return render(request, 'adminpanel/upload_video.html')
