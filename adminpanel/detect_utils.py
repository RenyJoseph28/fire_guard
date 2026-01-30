import cv2
import os
from ultralytics import YOLO
from django.conf import settings
from .models import Alert
from django.core.files.base import ContentFile
import io
from PIL import Image
import numpy as np

class FireDetector:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Path to your custom trained model
            # Ensure you put your best.pt file in the root fire_guard directory
            model_path = os.path.join(settings.BASE_DIR, 'best.pt') 
            if os.path.exists(model_path):
                cls._model = YOLO(model_path)
            else:
                print(f"Warning: Model not found at {model_path}. Using standard yolo11n.pt for demo.")
                cls._model = YOLO('yolo11n.pt') # Fallback
        return cls._model

    @staticmethod
    def process_video(video_path):
        """
        Process a video file, detect fire/smoke, and save alerts.
        """
        model = FireDetector.get_model()
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        skip_frames = 30  # Analyze every 30th frame (1 sec approx) to save resources
        alerts_created = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Run inference
            results = model(frame)
            
            for result in results:
                # Check detections
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[class_id]
                    
                    # Customize these labels based on your trained model
                    # For standard YOLO (coco), 'fire' isn't a class, but assuming your custom model:
                    target_classes = ['fire', 'smoke', 'burning']
                    
                    # If using standard YOLO, let's just log everything for now
                    # But for your specific request:
                    if label.lower() in target_classes or conf > 0.5: 
                         # (Adjust logic: if strictly your model, check label in target_classes)
                         
                         # Save Alert
                         FireDetector.save_alert(frame, label, conf)
                         alerts_created += 1
        
        cap.release()
        return alerts_created

    @staticmethod
    def save_alert(frame, label, confidence):
        # Convert frame to image file for storage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        img_io = io.BytesIO()
        pil_img.save(img_io, format='JPEG', quality=70)
        img_content = ContentFile(img_io.getvalue(), name=f"alert_{label}.jpg")

        # Determine severity
        severity = 'medium'
        if label.lower() == 'fire':
            severity = 'high'
        elif label.lower() == 'smoke':
            severity = 'low'

        # Create DB record
        Alert.objects.create(
            alert_type=label,
            confidence=confidence,
            severity=severity,
            snapshot=img_content
        )
