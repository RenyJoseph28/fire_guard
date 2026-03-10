import cv2
import os
from ultralytics import YOLO
from django.conf import settings
from .models import Alert
from django.core.files.base import ContentFile
import io
from PIL import Image
import numpy as np

def detect_gas_cylinder_context(frame, fire_box, frame_width, frame_height):
    """
    Detect gas cylinder based on red cylindrical object below fire
    (From enhanced_fire_analysis.py)
    """
    x1, y1, x2, y2 = fire_box
    
    # Define search region - WIDER area, focus below fire
    search_x1 = max(0, x1 - 80)
    search_x2 = min(frame_width, x2 + 80)
    search_y1 = max(0, y2 - 50)
    search_y2 = min(frame_height, y2 + 250)
    
    roi = frame[search_y1:search_y2, search_x1:search_x2]
    if roi.size == 0:
        return False
    
    # Color analysis - VERY LENIENT for red/maroon/orange
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # RELAXED red detection - catches orange-red to dark red
    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 40])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Noise filtering
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Shape analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / (h + 1e-6)
            
            # 0.2 = very tall, 1.5 = wider than tall
            if 0.2 < aspect_ratio < 1.5:
                if w > 20 and h > 30:
                    return True
    return False

def is_valid_fire_smoke(frame, xyxy, label, conf, width, height, has_fire_in_frame=False):
    """
    Apply Smart Filter to distinguish Fire vs Sunlight and Smoke vs Fog/Clouds
    
    Args:
        has_fire_in_frame: If True, skip blue sky rejection for smoke (real fire can have smoke against sky)
    """
    x1, y1, x2, y2 = xyxy
    # Clamp coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width-1, x2), min(height-1, y2)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return False
    
    label_lower = label.lower()
    
    # 1. Fire vs Sunlight Filter
    if 'fire' in label_lower:
        avg_color = cv2.mean(roi)[:3] # BGR
        b, g, r_val = avg_color
        r_val = max(r_val, 1.0)
        
        rb_ratio = r_val / (max(b, 1.0))
        rg_ratio = r_val / (max(g, 1.0))
        max_val = max(r_val, g, b)
        
        # Texture check for Fire (Fire is turbulent)
        (mean, std_dev) = cv2.meanStdDev(roi)
        texture_score = np.mean(std_dev)
        
        sunlight_score = 0.0
        
        # Sunlight is balanced (White) or Yellow (R~G) and often smooth
        if rb_ratio < 1.5 and rg_ratio < 1.3:
            sunlight_score += 0.4
            if max_val > 220: sunlight_score += 0.4 # Bright white
            if texture_score < 20: sunlight_score += 0.3 # Smooth
        
        # Capping
        sunlight_score = min(sunlight_score, 1.0)
        
        # Dynamic Threshold: Require higher confidence if it looks like sunlight
        dynamic_thresh = 0.15 + (sunlight_score * 0.5)
        if conf < dynamic_thresh:
            return False

    # 2. Smoke vs Fog / Cloud Filter
    elif 'smoke' in label_lower or 'default' in label_lower:
        (mean, std_dev) = cv2.meanStdDev(roi)
        texture_score = np.mean(std_dev)
        avg_val = np.mean(mean) # Brightness
        
        # CRITICAL FIX: Dark smoke (like from house fires) should ALWAYS pass!
        if avg_val < 90:
            print(f"DEBUG: Dark smoke detected (brightness {avg_val:.0f}) - ALLOWING")
            return True
        
        # Check for BLUE SKY anywhere in frame
        b_val, g_val, r_val = cv2.mean(roi)[:3]
        
        sky_height = min(150, height // 4)
        sky_region = frame[0:sky_height, :]
        
        is_blue_sky = False
        if sky_region.size > 0:
            sky_b, sky_g, sky_r = cv2.mean(sky_region)[:3]
            if sky_b > 100 and sky_b > sky_r * 1.2:
                is_blue_sky = True
                print(f"DEBUG: Blue sky detected in frame (B={sky_b:.0f}, R={sky_r:.0f})")
        
        # KEY FIX: Only reject as clouds if NO FIRE in frame
        # Real fire CAN have smoke against blue sky (outdoor fires, building fires)
        if is_blue_sky and avg_val >= 100 and not has_fire_in_frame:
            print(f"DEBUG: Rejected as CLOUDS (brightness {avg_val:.0f} + blue sky + no fire)")
            return False
        elif is_blue_sky and avg_val >= 100 and has_fire_in_frame:
            print(f"DEBUG: Blue sky + smoke BUT fire is present - ALLOWING smoke")
            # Continue to fog score check, don't reject outright

        fog_score = 0.0
        
        # A. Uniformity Check: Fog/clouds are very smooth
        if texture_score < 15: 
            fog_score += 0.5
        elif texture_score < 25:
            fog_score += 0.3
            
        # B. Brightness Check: Fog/clouds are bright white/grey
        if avg_val > 180:
            fog_score += 0.4
        elif avg_val > 150:
            fog_score += 0.2
            
        # C. Size/Shape Check: Fog/clouds often cover large area
        box_w = x2 - x1
        if box_w > width * 0.5:
            fog_score += 0.3
        
        # D. Saturation Check: Fog/clouds are very unsaturated (grey/white)
        saturation = max(b_val, g_val, r_val) - min(b_val, g_val, r_val)
        if saturation < 20: 
            fog_score += 0.3
        
        # E. Blue sky nearby adds penalty
        if is_blue_sky:
            fog_score += 0.4

        # Max penalty
        fog_score = min(fog_score, 1.0)
        
        # Dynamic Threshold: LESS AGGRESSIVE for fog-like detections in live camera
        # If fog_score = 1.0, need confidence > 0.55 to pass (was previous >0.90)
        dynamic_thresh = 0.15 + (fog_score * 0.40) 
        
        if conf < dynamic_thresh:
            print(f"DEBUG: Rejected Fog candidate '{label}' (Conf {conf:.2f} < Thresh {dynamic_thresh:.2f}, Score {fog_score:.2f})")
            return False
            
    return True

class FireDetector:
    _fire_model = None
    _material_model = None

    @classmethod
    def get_fire_model(cls):
        if cls._fire_model is None:
            model_path = os.path.join(settings.BASE_DIR, 'best.pt')
            try:
                if os.path.exists(model_path):
                    cls._fire_model = YOLO(model_path)
                else:
                    print(f"Warning: Fire model not found at {model_path}. Using standard yolov8n.pt")
                    cls._fire_model = YOLO('yolov8n.pt')
            except Exception as e:
                print(f"ERROR: Failed to load Fire model ({model_path}): {e}")
                print("Fallback to standard yolov8n.pt")
                cls._fire_model = YOLO('yolov8n.pt')
        return cls._fire_model

    @classmethod
    def get_material_model(cls):
        if cls._material_model is None:
            model_path = os.path.join(settings.BASE_DIR, 'best_material.pt')
            try:
                if os.path.exists(model_path):
                    cls._material_model = YOLO(model_path)
                else:
                    print(f"Warning: Material model not found at {model_path}. Using standard yolov8n.pt")
                    cls._material_model = YOLO('yolov8n.pt')
            except Exception as e:
                print(f"ERROR: Failed to load Material model ({model_path}): {e}")
                print("Fallback to standard yolov8n.pt")
                cls._material_model = YOLO('yolov8n.pt')
        return cls._material_model

    @staticmethod
    def process_video(video_path):
        """
        Process a video file, detect fire/smoke and burning materials, and save alerts.
        """
        fire_model = FireDetector.get_fire_model()
        material_model = FireDetector.get_material_model()
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        skip_frames = 30  # Analyze every 30th frame
        alerts_created = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # --- Collect all detections for this frame ---
            frame_detections = []
            has_cv2_cylinder = False
            
            # TWO-PASS APPROACH:
            # Pass 1: Detect FIRE first (so we know if real fire exists in frame)
            # Pass 2: Process smoke (skip blue sky check if fire was found)
            
            has_fire_in_frame = False
            raw_detections = []
            
            # 1. Fire Model - First collect all raw detections
            if fire_model:
                results_fire = fire_model(frame, verbose=False)
                for result in results_fire:
                    for box in result.boxes:
                        label = fire_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        
                        raw_detections.append({
                            'label': label,
                            'conf': conf,
                            'xyxy': xyxy,
                            'source': 'fire_model'
                        })
                        
                        # Check if this is a fire detection (before filtering)
                        # 'default' label in this model represents fire/burning
                        if ('fire' in label.lower() or label.lower() == 'default') and conf > 0.15:
                            has_fire_in_frame = True
            
            # Pass 2: Now filter detections, but with context of fire presence
            for det in raw_detections:
                label = det['label']
                conf = det['conf']
                xyxy = det['xyxy']
                
                # Apply smart filter with fire context
                if not is_valid_fire_smoke(frame, xyxy, label, conf, width, height, has_fire_in_frame):
                    continue
                
                frame_detections.append({
                    'label': label,
                    'conf': conf,
                    'source': det['source']
                })
                
                # Run CV2 Heuristic for cylinders near fire
                if 'fire' in label.lower() and conf > 0.15:
                     if detect_gas_cylinder_context(frame, xyxy, width, height):
                         has_cv2_cylinder = True
                         print(f"DEBUG: CV2 Heuristic detected Gas Cylinder near {label}!")

            # 2. Material Model (YOLO) - Detect objects that could be burning materials
            detected_materials = []
            if material_model:
                results_material = material_model(frame, verbose=False)
                for result in results_material:
                    for box in result.boxes:
                        label = material_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        
                        if conf > 0.25:  # Only consider confident detections
                            detected_materials.append(label.lower())
                            frame_detections.append({
                                'label': label,
                                'conf': conf,
                                'source': 'material_model'
                            })
            
            # --- Analyze Context ---
            # Note: 'default' label in this model represents fire/burning material
            has_fire = any(
                ('fire' in d['label'].lower() or d['label'].lower() == 'default') 
                and d['conf'] > 0.15 
                for d in frame_detections
            )
            
            # Check for cylinder from YOLO OR CV2
            has_yolo_cylinder = any(('cylinder' in d['label'].lower() or 'gas' in d['label'].lower()) and d['conf'] > 0.15 for d in frame_detections)
            has_cylinder = has_yolo_cylinder or has_cv2_cylinder
            
            # --- MATERIAL-BASED FIRE CLASSIFICATION & EXPLOSION RISK ---
            fire_class = None
            fire_class_desc = None
            explosion_risk = False
            burning_materials = []
            
            # COCO object mapping to fire classes and materials
            # These mappings help identify what's burning
            material_mapping = {
                # Class B - Flammable Liquids (EXPLOSION RISK!)
                'bottle': ('B', 'Flammable Container', True),
                'car': ('B', 'Vehicle/Fuel', True),
                'truck': ('B', 'Vehicle/Fuel', True),
                'motorcycle': ('B', 'Vehicle/Fuel', True),
                'bus': ('B', 'Vehicle/Fuel', True),
                'airplane': ('B', 'Aircraft/Fuel', True),
                
                # Class C - Electrical Equipment
                'laptop': ('C', 'Electronics', False),
                'tv': ('C', 'Electronics', False),
                'cell phone': ('C', 'Electronics', False),
                'microwave': ('C', 'Appliance', False),
                'toaster': ('C', 'Appliance', False),
                'refrigerator': ('C', 'Appliance', False),
                
                # Class K - Kitchen/Cooking
                'oven': ('K', 'Cooking Equipment', True),  # Grease fire risk
                'bowl': ('K', 'Kitchen Item', False),
                'cup': ('K', 'Kitchen Item', False),
                'fork': ('K', 'Kitchen Item', False),
                'knife': ('K', 'Kitchen Item', False),
                'spoon': ('K', 'Kitchen Item', False),
                
                # Class A - Ordinary Combustibles
                'book': ('A', 'Paper/Books', False),
                'couch': ('A', 'Furniture', False),
                'bed': ('A', 'Furniture', False),
                'chair': ('A', 'Furniture', False),
                'dining table': ('A', 'Furniture', False),
                'potted plant': ('A', 'Organic Material', False),
                'backpack': ('A', 'Fabric/Textile', False),
                'handbag': ('A', 'Fabric/Textile', False),
                'suitcase': ('A', 'Fabric/Textile', False),
                'teddy bear': ('A', 'Fabric/Textile', False),
            }
            
            # Analyze detected materials
            for mat in detected_materials:
                if mat in material_mapping:
                    cls, desc, risk = material_mapping[mat]
                    burning_materials.append(f"{mat.title()} ({desc})")
                    if fire_class is None or (cls == 'B' and fire_class != 'B'):
                        fire_class = cls
                        fire_class_desc = desc
                    if risk:
                        explosion_risk = True
            
            # Also check for cylinder (highest explosion risk)
            if has_cylinder:
                fire_class = 'B'
                fire_class_desc = 'Gas Cylinder'
                explosion_risk = True
                burning_materials.append("Gas Cylinder (EXTREME DANGER)")
            
            # Default if fire detected but no specific material
            if has_fire and fire_class is None:
                fire_class = 'A'
                fire_class_desc = 'General Fire'
            
            # Determine logic
            processed_labels = set()
            
            # Log detected materials (if any)
            if burning_materials and has_fire:
                materials_str = ", ".join(burning_materials[:3])  # Top 3 materials
                print(f"DEBUG: Burning materials detected: {materials_str}")
            
            # CRITICAL: Explosion Risk Alert
            if has_fire and explosion_risk:
                if has_cylinder:
                    alert_label = "EXPLOSION RISK - Gas/Flammable near Fire"
                else:
                    alert_label = f"CLASS {fire_class} - {fire_class_desc}"
                FireDetector.save_alert(frame, alert_label, 0.99, severity='critical')
                alerts_created += 1
                processed_labels.add('explosion_risk')
                print(f"DEBUG: ⚠️ EXPLOSION RISK DETECTED!")
            # Other fire classes (no explosion risk)
            elif has_fire and fire_class:
                class_label = f"CLASS {fire_class} - {fire_class_desc}"
                # Class B, C, K are more dangerous
                sev = 'critical' if fire_class in ['B', 'C', 'K'] else 'high'
                FireDetector.save_alert(frame, class_label, 0.95, severity=sev)
                alerts_created += 1
                processed_labels.add('fire_class')
            
            # Log detected materials as separate alerts (for tracking)
            if burning_materials and has_fire and len(burning_materials) > 0:
                # Create one consolidated material alert
                materials_alert = "Materials: " + ", ".join(burning_materials[:3])
                if len(materials_alert) > 95:  # Truncate if too long
                    materials_alert = materials_alert[:92] + "..."
                FireDetector.save_alert(frame, materials_alert, 0.80, severity='medium')
                alerts_created += 1
            
            # Log other (relevant) detections
            allowed_labels = ['fire', 'smoke', 'cylinder', 'gas', 'burning', 'default', 'flames']
            
            for d in frame_detections:
                label = d['label']
                conf = d['conf']
                
                if conf < 0.15: continue
                
                # STRICT FILTER: Only allow relevant fire/safety classes
                if label.lower() not in allowed_labels:
                    continue
                
                # Filter redundant if we already created a class-specific alert
                is_handled = 'fire_class' in processed_labels or 'class_b' in processed_labels
                if 'fire' in label.lower() and is_handled: continue
                if ('cylinder' in label.lower() or 'gas' in label.lower()) and 'class_b' in processed_labels: continue
                
                print(f"DEBUG: Detected {label} with confidence {conf}")
                FireDetector.save_alert(frame, label, conf)
                alerts_created += 1
        
        cap.release()
        return alerts_created

    @staticmethod
    def process_live_frame(frame):
        """Processes a single live frame, returning a list of detections and ensuring throttled alerts."""
        fire_model = FireDetector.get_fire_model()
        width = frame.shape[1]
        height = frame.shape[0]
        
        frame_detections = []
        has_fire_in_frame = False
        raw_detections = []
        alerts_created = 0
        detections_for_ui = []
        
        if fire_model:
            results_fire = fire_model(frame, verbose=False, conf=0.1)
            for result in results_fire:
                for box in result.boxes:
                    label = fire_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    raw_detections.append({
                        'label': label,
                        'conf': conf,
                        'xyxy': xyxy
                    })
                    
                    if ('fire' in label.lower() or label.lower() == 'default') and conf > 0.15:
                        has_fire_in_frame = True
        
        for det in raw_detections:
            label = det['label']
            conf = det['conf']
            xyxy = det['xyxy']
            
            if not is_valid_fire_smoke(frame, xyxy, label, conf, width, height, has_fire_in_frame):
                continue
            
            display_label = label if label.lower() != 'default' else 'Fire'
            
            detections_for_ui.append({
                'label': display_label,
                'conf': float(conf),
                'box': [int(x) for x in xyxy]
            })
            
            allowed_labels = ['fire', 'smoke', 'cylinder', 'gas', 'burning', 'default', 'flames']
            if label.lower() in allowed_labels and conf > 0.15:
                # Throttling logic! We don't want 30 emails per second.
                from django.utils import timezone
                from datetime import timedelta
                
                # Check for recent identical alerts
                alert_type_name = label if label.lower() != 'default' else 'Fire/Burning Detected'
                
                recent_alerts = Alert.objects.filter(alert_type=alert_type_name).order_by('-timestamp')
                should_alert = True
                if recent_alerts.exists():
                    latest = recent_alerts.first()
                    # Alert once every 20s
                    if timezone.now() - latest.timestamp < timedelta(seconds=20):
                        should_alert = False
                
                if should_alert:
                    FireDetector.save_alert(frame, label, conf)
                    alerts_created += 1
                    
        return detections_for_ui, alerts_created

    @staticmethod
    def save_alert(frame, label, confidence, severity=None):
        # Convert frame to image for storage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        img_io = io.BytesIO()
        pil_img.save(img_io, format='JPEG', quality=70)
        
        # Sanitize filename to remove invalid characters (like :) for Windows
        safe_label = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label]).strip().replace(' ', '_')
        img_content = ContentFile(img_io.getvalue(), name=f"alert_{safe_label}.jpg")

        # Determine severity if not forced
        if severity is None:
            severity = 'medium'
            label_lower = label.lower()
            
            # 'default' is often used as class name for fire in custom models
            if 'fire' in label_lower or 'default' in label_lower:
                severity = 'high'
            elif 'class b' in label_lower:
                severity = 'critical'
            elif 'cylinder' in label_lower or 'gas' in label_lower:
                severity = 'critical'
            elif 'smoke' in label_lower:
                severity = 'high'  # Changed from 'low' - smoke is also serious!
            elif 'burning' in label_lower:
                severity = 'high'
        
        # Rename 'default' to something more user-friendly
        display_label = label
        if label.lower() == 'default':
            display_label = 'Fire/Burning Detected'

        # Avoid duplicates: Check if similar alert exists recently (optional logic)
        # Avoid duplicates: Check if similar alert exists recently (optional logic)
        alert = Alert.objects.create(
            alert_type=display_label,
            confidence=confidence,
            severity=severity,
            snapshot=img_content
        )
        
        # Send Email Notification
        try:
            FireDetector.send_alert_email(alert)
        except Exception as e:
            print(f"Error sending email alert: {e}")
        
        # Send Push Notification
        try:
            from .views import send_push_notification
            title = f"🔥 {display_label}"
            body = f"Severity: {severity.upper()} | Location: Camera 1"
            send_push_notification(title, body, {'alert_id': alert.id})
        except Exception as e:
            print(f"Error sending push notification: {e}")

    @staticmethod
    def send_alert_email(alert):
        from django.core.mail import EmailMultiAlternatives
        from django.conf import settings
        from .models import AlertRecipient

        # 1. Get recipients
        is_critical = alert.severity in ['high', 'critical']
        if is_critical:
            recipients = AlertRecipient.objects.filter(is_active=True)
        else:
            recipients = AlertRecipient.objects.filter(is_active=True, receive_critical_only=False)
            
        if not recipients.exists():
            return

        recipient_list = [r.email for r in recipients]
        
        # 2. Construct Email
        # Subject
        icon = "🔥"
        if alert.severity == 'critical': icon = "⚠️"
        elif alert.severity == 'low': icon = "☁️"
        
        subject = f"{icon} FIRE GUARD ALERT: {alert.alert_type} Detected [{alert.severity.upper()}]"
        
        # Plain Text Body (Fallback)
        text_body = f"""
        FIRE GUARD SECURITY ALERT
        =========================
        
        A potential hazard has been detected by the AI Surveillance System.
        
        DETAILS:
        - Hazard Type: {alert.alert_type}
        - Severity: {alert.severity.upper()}
        - Location: {alert.location}
        - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        
        ACTION REQUIRED:
        Please verify this alert immediately. Check the FireGuard Admin Dashboard for live feed and analysis.
        
        --
        Automated Alert System - Fire Guard
        """
        
        # HTML Body (Nice looking)
        color = "#e11d48" # Red
        if alert.severity == 'medium': color = "#f97316" # Orange
        if alert.severity == 'low': color = "#6b7280" # Grey
        
        # Determine functionality based on class
        extinguisher = "General Safety Protocols"
        
        at_upper = alert.alert_type.upper()
        if "CLASS A" in at_upper:
            extinguisher = "Water, Foam, ABC Powder"
        elif "CLASS B" in at_upper or "LIQUID" in at_upper or "GAS" in at_upper or "CYLINDER" in at_upper:
            extinguisher = "Foam, CO2, Dry Powder"
        elif "CLASS C" in at_upper or "ELECTRICAL" in at_upper:
            extinguisher = "CO2, Dry Powder (DO NOT USE WATER)"
        elif "CLASS K" in at_upper or "COOKING" in at_upper:
            extinguisher = "Wet Chemical"
            
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 25px; background-color: #ffffff; }}
                .alert-box {{ background-color: #fff1f2; border-left: 4px solid {color}; padding: 15px; margin: 20px 0; }}
                .details-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .details-table td {{ padding: 8px 0; border-bottom: 1px solid #f3f4f6; }}
                .label {{ font-weight: bold; color: #555; width: 30%; }}
                .footer {{ background-color: #f9fafb; padding: 15px; text-align: center; font-size: 12px; color: #6b7280; }}
                .button {{ display: inline-block; padding: 10px 20px; background-color: {color}; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin:0; font-size: 24px;">SECURITY ALERT</h1>
                </div>
                <div class="content">
                    <h2 style="margin-top:0; color: {color};">⚠️ {alert.alert_type} Detected</h2>
                    <p>The FireGuard AI System has detected a potential fire hazard requiring your attention.</p>
                    
                    <div class="alert-box">
                        <table class="details-table">
                            <tr>
                                <td class="label">Hazard Type:</td>
                                <td><strong>{alert.alert_type}</strong></td>
                            </tr>
                            <tr>
                                <td class="label">Severity Level:</td>
                                <td style="color: {color}; font-weight: bold;">{alert.severity.upper()}</td>
                            </tr>
                            <tr>
                                <td class="label">Recommended Extinguisher:</td>
                                <td style="color: #d97706; font-weight: bold;">{extinguisher}</td>
                            </tr>
                            <tr>
                                <td class="label">Location:</td>
                                <td>{alert.location}</td>
                            </tr>
                            <tr>
                                <td class="label">Time:</td>
                                <td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <p><strong>Recommended Action:</strong> Please verify the situation immediately. If this is a real emergency, initiate standard safety protocols.</p>
                    
                    <center>
                        <p style="font-size: 14px; color: #888;">Snapshot from surveillance feed is attached below.</p>
                    </center>
                </div>
                <div class="footer">
                    &copy; 2025 Fire Guard AI System. Automated Message.<br>
                    Please do not reply to this email.
                </div>
            </div>
        </body>
        </html>
        """
        
        email = EmailMultiAlternatives(
            subject,
            text_body,
            settings.DEFAULT_FROM_EMAIL,
            [],  # To
            recipient_list,  # Bcc
        )
        # Attach HTML version
        email.attach_alternative(html_body, "text/html")
        
        # 3. Attach Snapshot
        if alert.snapshot:
            try:
                alert.snapshot.open('rb')
                email.attach(alert.snapshot.name, alert.snapshot.read(), 'image/jpeg')
                alert.snapshot.close()
            except Exception as e:
                print(f"Could not attach image: {e}")
        
        # 4. Send
        print(f"Sending email alert to {len(recipient_list)} recipients...")
        email.send(fail_silently=False)
        print("Email sent successfully.")
