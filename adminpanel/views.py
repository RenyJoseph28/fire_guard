from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Alert
from django.core.files.storage import FileSystemStorage
from .detect_utils import FireDetector
import os
import base64
import numpy as np
import cv2
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


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
    
    # Calculate stats
    total_users = User.objects.filter(is_superuser=False).count()
    active_alerts = Alert.objects.filter(is_resolved=False).count()
    
    context = {
        'recent_alerts': recent_alerts,
        'total_users': total_users,
        'active_alerts': active_alerts,
        'alert_count': Alert.objects.count(),
        'high_severity_count': Alert.objects.filter(severity__in=['high', 'critical']).count()
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

from .models import AlertRecipient

def recipients_list(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')
    
    recipients = AlertRecipient.objects.all().order_by('-created_at')
    return render(request, 'adminpanel/recipients_list.html', {'recipients': recipients})

def add_recipient(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')
        
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        critical_only = request.POST.get('critical_only') == 'on'
        
        if AlertRecipient.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
        else:
            AlertRecipient.objects.create(
                name=name,
                email=email,
                receive_critical_only=critical_only
            )
            messages.success(request, "Recipient added successfully.")
            return redirect('adminpanel:recipients_list')
            
    return render(request, 'adminpanel/add_recipient.html')

def delete_recipient(request, recipient_id):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')
        
    try:
        recipient = AlertRecipient.objects.get(id=recipient_id)
        recipient.delete()
        messages.success(request, "Recipient removed.")
    except AlertRecipient.DoesNotExist:
        messages.error(request, "Recipient not found.")
        
    return redirect('adminpanel:recipients_list')

def live_detection(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('adminpanel:admin_login')
    return render(request, 'adminpanel/live_detection.html')

@csrf_exempt
@require_POST
def process_live_frame(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        from django.http import JsonResponse
        return JsonResponse({'error': 'Unauthorized'}, status=401)
        
    try:
        from django.http import JsonResponse
        import json
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)
            
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Invalid image format'}, status=400)
            
        # Use our new static method
        detections, alerts_created = FireDetector.process_live_frame(frame)
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'alerts_created': alerts_created
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        from django.http import JsonResponse
        return JsonResponse({'error': str(e)}, status=500)

# ========== FCM Push Notifications ==========
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import FCMDevice
import json
import requests

@csrf_exempt
@require_POST
def save_fcm_token(request):
    """API endpoint to save FCM token from frontend"""
    try:
        data = json.loads(request.body)
        token = data.get('token')
        device_name = data.get('device_name', 'Unknown Device')
        
        if not token:
            return JsonResponse({'error': 'Token is required'}, status=400)
        
        # Create or update the device
        device, created = FCMDevice.objects.update_or_create(
            token=token,
            defaults={
                'user': request.user if request.user.is_authenticated else None,
                'device_name': device_name,
                'is_active': True
            }
        )
        
        print(f"FCM Token saved: {device_name} ({'new' if created else 'updated'})")
        return JsonResponse({'success': True, 'created': created, 'device_name': device_name})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def send_push_notification(title, body, data=None):
    """
    Send push notification to all registered FCM devices.
    Uses Firebase Admin SDK with Service Account.
    Includes throttling to avoid notification flooding.
    """
    from django.conf import settings
    import firebase_admin
    from firebase_admin import credentials, messaging
    import time
    
    # Throttle: only send one notification every 30 seconds
    current_time = time.time()
    last_sent = getattr(send_push_notification, '_last_sent_time', 0)
    if current_time - last_sent < 30:
        print(f"Push notification throttled (sent {int(current_time - last_sent)}s ago, wait 30s)")
        return False
    
    # Get all active FCM tokens
    devices = FCMDevice.objects.filter(is_active=True)
    tokens = [d.token for d in devices]
    
    if not tokens:
        print("No FCM devices registered")
        return False
    
    # Initialize Firebase Admin SDK (only once)
    if not firebase_admin._apps:
        cred_path = getattr(settings, 'FIREBASE_CREDENTIALS_FILE', None)
        if not cred_path:
            print("FIREBASE_CREDENTIALS_FILE not configured in settings.py")
            return False
        
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
    
    # Create message for each token (FCM V1 API)
    success_count = 0
    failure_count = 0
    
    for token in tokens:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            webpush=messaging.WebpushConfig(
                notification=messaging.WebpushNotification(
                    icon='/static/icons/icon-192x192.png',
                    badge='/static/icons/icon-72x72.png',
                    vibrate=[500, 200, 500, 200, 500, 200, 500],
                    require_interaction=True,
                ),
            ),
            # High priority for Android to wake the device and play sound
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    sound='default',
                    priority='max',
                    channel_id='fire_alerts',
                    notification_count=1,
                    sticky=True,
                ),
            ),
            data={str(k): str(v) for k, v in (data or {}).items()},
            token=token
        )
        
        try:
            response = messaging.send(message)
            print(f"FCM Success: {response}")
            success_count += 1
        except Exception as e:
            error_str = str(e).lower()
            print(f"FCM Error for token {token[:20]}...: {e}")
            failure_count += 1
            # Deactivate invalid/unregistered tokens
            if 'notregistered' in error_str or 'not-registered' in error_str or 'not registered' in error_str or 'invalid' in error_str:
                FCMDevice.objects.filter(token=token).update(is_active=False)
                print(f"  → Token deactivated")
    
    # Update last sent time
    send_push_notification._last_sent_time = current_time
    
    print(f"Push notifications sent: {success_count} success, {failure_count} failed")
    return success_count > 0

