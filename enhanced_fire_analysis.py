#!/usr/bin/env python3
"""
ENHANCED INTEGRATED FIRE ANALYSIS SYSTEM
Combines:
- Smart fire/smoke detection with color filtering (from run_detection.py)
- Material classification (Class A, B, C, D, K)
- Gas cylinder detection
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import os

class MaterialClassifier:
    """ResNet-based material classifier for fire types"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize material classifier"""
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        
        # Fire class names
        self.class_names = {
            0: 'Class A',  # Wood/Paper/Fabric
            1: 'Class B',  # Flammable Liquids
            2: 'Class C',  # Electrical
            3: 'Class D',  # Combustible Metals
            4: 'Class K'   # Cooking Oils/Grease
        }
        
        self.class_descriptions = {
            0: 'Wood/Paper/Fabric',
            1: 'Flammable Liquids',
            2: 'Electrical',
            3: 'Combustible Metals',
            4: 'Cooking Oils/Grease'
        }
        
        # Build model
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 5)
        
        # Load weights
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Predict fire class from image"""
        if image.size == 0:
            return None, None, 0.0
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Transform
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_id = predicted.item()
        class_name = self.class_names[class_id]
        description = self.class_descriptions[class_id]
        conf = confidence.item()
        
        return class_id, class_name, description, conf


def detect_gas_cylinder_context(frame, fire_box, frame_width, frame_height):
    """
    Detect gas cylinder based on red cylindrical object below fire
    RELAXED VERSION - More sensitive detection
    """
    x1, y1, x2, y2 = fire_box
    
    # Define search region - WIDER area, focus below fire
    search_x1 = max(0, x1 - 80)  # Wider search
    search_x2 = min(frame_width, x2 + 80)
    search_y1 = max(0, y2 - 50)  # Start a bit above fire bottom
    search_y2 = min(frame_height, y2 + 250)  # Look deeper below
    
    roi = frame[search_y1:search_y2, search_x1:search_x2]
    if roi.size == 0:
        return False
    
    # Color analysis - VERY LENIENT for red/maroon/orange
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # RELAXED red detection - catches orange-red to dark red
    lower_red1 = np.array([0, 50, 40])   # Very low thresholds
    upper_red1 = np.array([15, 255, 255]) # Extended to orange
    lower_red2 = np.array([165, 50, 40])  # Extended range
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Less aggressive noise filtering
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Shape analysis - VERY LENIENT
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Much lower area requirement
        if area > 500:  # Was 800
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / (h + 1e-6)
            
            # Very wide range - catches cylinders at any angle
            # 0.2 = very tall, 1.5 = wider than tall (horizontal view)
            if 0.2 < aspect_ratio < 1.5:
                # Very relaxed size requirements
                if w > 20 and h > 30:  # Was w>25, h>40
                    return True
    
    return False


class EnhancedFireAnalysis:
    """Enhanced fire analysis with smart filtering + material classification"""
    
    def __init__(self, fire_model_path, material_model_path=None, base_conf=0.15):
        """Initialize the system"""
        
        print("="*70)
        print("ENHANCED INTEGRATED FIRE ANALYSIS SYSTEM")
        print("="*70)
        
        # Load fire detector
        print("📦 Loading fire/smoke detector...")
        print(f"   Model: {fire_model_path}")
        self.fire_detector = YOLO(fire_model_path)
        self.base_conf = base_conf
        print("   ✅ Fire detector loaded")
        print(f"   Base confidence: {base_conf}")
        print("   Smart filtering: ENABLED")
        
        # Load material classifier
        self.material_classifier = None
        if material_model_path and Path(material_model_path).exists():
            print("📦 Loading material classifier...")
            print(f"   Model: {material_model_path}")
            try:
                self.material_classifier = MaterialClassifier(material_model_path)
                print("   ✅ Material classifier loaded")
            except Exception as e:
                print(f"   ⚠️  Failed to load: {e}")
        else:
            print("⚠️  Material classifier not available")
        
        print("="*70)
        print("✅ SYSTEM READY")
        print("="*70)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'fire_frames': 0,
            'smoke_frames': 0,
            'material_counts': {},
            'gas_cylinder_detections': 0
        }
    
    def apply_smart_filter(self, frame, box, class_id, confidence, frame_width, frame_height):
        """
        Apply smart filtering to reduce false positives
        (From your run_detection.py - PROVEN TO WORK!)
        
        Returns: (is_valid, dynamic_threshold, filter_score)
        """
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width-1, x2), min(frame_height-1, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, self.base_conf, 0.0
        
        # FIRE/BURNING MATERIAL FILTER
        if class_id in [0, 1]:  # Fire or burning material
            avg_color = cv2.mean(roi)[:3]
            b, g, r_val = avg_color
            r_val = max(r_val, 1.0)
            
            rb_ratio = r_val / max(b, 1.0)
            rg_ratio = r_val / max(g, 1.0)
            max_val = max(r_val, g, b)
            
            # Texture check
            (mean, std_dev) = cv2.meanStdDev(roi)
            texture_score = np.mean(std_dev)
            
            sunlight_score = 0.0
            
            # Detect sunlight (balanced white/yellow, smooth)
            if rb_ratio < 1.15 and rg_ratio < 1.1:
                sunlight_score += 0.3
                if max_val > 240:
                    sunlight_score += 0.5
            
            if texture_score < 20:
                sunlight_score += 0.3
            
            sunlight_score = min(sunlight_score, 1.0)
            dynamic_thresh = self.base_conf + (sunlight_score * 0.5)
            
            return confidence >= dynamic_thresh, dynamic_thresh, sunlight_score
        
        # SMOKE FILTER (Fog detection)
        elif class_id == 2:
            (mean, std_dev) = cv2.meanStdDev(roi)
            texture_score = np.mean(std_dev)
            avg_val = np.mean(mean)
            
            fog_score = 0.0
            
            # Uniformity
            if texture_score < 20:
                fog_score += 0.5
            elif texture_score < 35:
                fog_score += 0.3
            
            # Brightness
            if avg_val > 170:
                fog_score += 0.2
            
            # Size/shape
            box_w = x2 - x1
            box_h = y2 - y1
            aspect_ratio = box_w / (box_h + 1e-6)
            
            if box_w > frame_width * 0.6:
                fog_score += 0.5
            if aspect_ratio > 1.3:
                fog_score += 0.2
            
            # Saturation
            b, g, r_v = cv2.mean(roi)[:3]
            saturation = max(b, g, r_v) - min(b, g, r_v)
            if saturation < 25:
                fog_score += 0.3
            
            fog_score = min(fog_score, 1.0)
            dynamic_thresh = self.base_conf + (fog_score * 0.8)
            
            return confidence >= dynamic_thresh, dynamic_thresh, fog_score
        
        return True, self.base_conf, 0.0
    
    def process_video(self, video_path, output_path=None):
        """Process video with smart filtering + material classification"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n🎥 Processing video...")
        print(f"   Source: {video_path}")
        print(f"   Total frames: {total_frames}")
        print(f"   Smart filtering: ACTIVE")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.stats['total_frames'] = frame_count
            
            # Run detection with LOW confidence (let smart filter handle it)
            results = self.fire_detector(frame, conf=0.05, verbose=False)
            
            has_fire = False
            has_smoke = False
            gas_cylinder_in_frame = False  # Track per frame
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Apply smart filter
                    is_valid, thresh, filter_score = self.apply_smart_filter(
                        frame, xyxy, class_id, confidence, width, height
                    )
                    
                    if not is_valid:
                        continue
                    
                    x1, y1, x2, y2 = xyxy
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)
                    
                    # Get label and check class
                    label = result.names[class_id]
                    
                    # DEBUG: Print what class is detected
                    if frame_count % 60 == 0:
                        print(f"   Frame {frame_count}: Detected '{label}' (class_id={class_id}) conf={confidence:.2f}")
                    
                    # Material classification and cylinder check
                    material_info = None
                    is_gas_cylinder = False
                    
                    # Check if this is a FIRE-RELATED detection
                    # Your model has: class_id 0 = 'Fire', class_id 1 = 'default' (smoke/burning)
                    is_fire_or_burning = (
                        class_id == 0 or  # Fire
                        class_id == 1 or  # Default (treat as fire/burning)
                        label.lower() in ['fire', 'burning', 'burning material', 'flames', 'burning_material', 'default', 'smoke']
                    )
                    
                    # Determine what type of detection this is
                    if class_id == 0 or label.lower() == 'fire':
                        detection_type = 'fire'
                    elif class_id == 1 or label.lower() == 'default':
                        detection_type = 'burning_or_smoke'  # Your class 1
                    else:
                        detection_type = 'smoke'
                    
                    if is_fire_or_burning:
                        has_fire = True
                        
                        # Check for gas cylinder
                        is_gas_cylinder = detect_gas_cylinder_context(
                            frame, (x1, y1, x2, y2), width, height
                        )
                        
                        # DEBUG: Print cylinder detection with details
                        if frame_count % 30 == 0:  # Print every 30 frames
                            fire_width = x2 - x1
                            fire_height = y2 - y1
                            print(f"   Frame {frame_count}: Cylinder={is_gas_cylinder}, Fire box: {fire_width}x{fire_height}px at y={y2}")
                        
                        if is_gas_cylinder and not gas_cylinder_in_frame:
                            gas_cylinder_in_frame = True
                            self.stats['gas_cylinder_detections'] += 1
                        
                        # Classify material
                        if self.material_classifier:
                            roi = frame[y1:y2, x1:x2]
                            class_id_mat, class_name, description, mat_conf = self.material_classifier.predict(roi)
                            
                            # OVERRIDE: If gas cylinder detected, force Class B
                            if is_gas_cylinder:
                                class_id_mat = 1
                                class_name = "Class B"
                                description = "Flammable Liquids (Gas Cylinder)"
                                mat_conf = 0.95  # High confidence override
                            
                            if class_name:
                                material_info = {
                                    'class': class_name,
                                    'description': description,
                                    'confidence': mat_conf
                                }
                                
                                # PRINT MATERIAL CLASSIFICATION TO TERMINAL
                                if is_gas_cylinder:
                                    print(f"   🔥 Frame {frame_count}: {class_name} - {description} [GAS CYLINDER DETECTED] (confidence: {mat_conf:.2%})")
                                else:
                                    print(f"   🔥 Frame {frame_count}: {class_name} - {description} (confidence: {mat_conf:.2%})")
                                
                                # Update statistics
                                key = f"{class_name} ({description})"
                                self.stats['material_counts'][key] = self.stats['material_counts'].get(key, 0) + 1
                    
                    # Check if smoke (if you have a separate smoke class)
                    elif class_id == 2 or label.lower() == 'smoke':
                        has_smoke = True
                    
                    # Draw detection with proper labels
                    if is_fire_or_burning:
                        # Determine display label based on detection type
                        if detection_type == 'fire':
                            display_label = "Fire"
                            color = (0, 0, 255)  # Red
                        elif detection_type == 'burning_or_smoke':
                            # This is your 'default' class - could be burning material or smoke
                            display_label = "Burning Material / Smoke"
                            color = (0, 140, 255)  # Orange
                        else:
                            display_label = label
                            color = (0, 140, 255)
                        
                        # Override if gas cylinder detected
                        if is_gas_cylinder:
                            display_label = f"Fire (Gas Cylinder)"
                            color = (0, 0, 255)  # Red for Class B
                    else:  # Smoke
                        display_label = "Smoke"
                        color = (128, 128, 128)  # Grey
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    text_lines = [f"{display_label} {confidence:.2f}"]
                    
                    if material_info:
                        text_lines.append(f"{material_info['class']}")
                        text_lines.append(f"{material_info['description']}")
                    
                    # Draw label background
                    y_offset = y1 - 10
                    for line in text_lines:
                        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        if y_offset < th:
                            y_offset = y1 + 20
                        cv2.rectangle(frame, (x1, y_offset - th - 5), (x1 + tw, y_offset), color, -1)
                        cv2.putText(frame, line, (x1, y_offset - 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += th + 5
            
            # Update stats
            if has_fire:
                self.stats['fire_frames'] += 1
            if has_smoke:
                self.stats['smoke_frames'] += 1
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display
            cv2.imshow('Enhanced Fire Analysis', frame)
            
            # Progress
            if frame_count % 60 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames...", end='\r')
            
            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        self.print_summary(output_path)
    
    def print_summary(self, output_path=None):
        """Print analysis summary"""
        print("\n\n" + "="*70)
        print("📊 ANALYSIS SUMMARY")
        print("="*70)
        print(f"✅ Processing complete!")
        print(f"   Total frames: {self.stats['total_frames']}")
        print(f"   Fire detected in: {self.stats['fire_frames']} frames")
        print(f"   Smoke detected in: {self.stats['smoke_frames']} frames")
        
        if self.stats['gas_cylinder_detections'] > 0:
            print(f"   ⚠️  Gas cylinder fires: {self.stats['gas_cylinder_detections']} detections")
        
        if self.stats['material_counts']:
            print(f"\n🔥 MATERIAL CLASSIFICATION RESULTS:")
            sorted_materials = sorted(
                self.stats['material_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for material, count in sorted_materials:
                print(f"   {material}: {count} detections")
            
            # Show recommendation for primary fire type
            primary_material = sorted_materials[0][0]
            print(f"\n⚠️  PRIMARY FIRE TYPE: {primary_material}")
            print(self.get_safety_recommendation(primary_material))
        
        if output_path:
            print(f"\n💾 Annotated video saved to: {output_path}")
        
        print("="*70)
    
    def get_safety_recommendation(self, material):
        """Get safety recommendations"""
        recommendations = {
            'Class A': '🧯 Use: Water, foam, or ABC extinguisher\n   ⚠️  Danger: Can spread through materials',
            'Class B': '🧯 Use: CO2, foam, or BC extinguisher (NEVER water)\n   ⚠️  Danger: Water can spread fire',
            'Class C': '🧯 Use: CO2 or dry chemical (NEVER water)\n   ⚠️  Danger: Electrical shock hazard',
            'Class D': '🧯 Use: Specialized dry powder ONLY\n   ⚠️  Danger: EXTREME - reacts with water',
            'Class K': '🧯 Use: Wet chemical or Class K extinguisher\n   ⚠️  Danger: NEVER use water - explosive splatter'
        }
        
        for key, rec in recommendations.items():
            if key in material:
                return rec
        
        return '🧯 Use: ABC extinguisher\n   ⚠️  Evacuate if unsure'


def main():
    parser = argparse.ArgumentParser(description='Enhanced Fire Analysis System')
    parser.add_argument('--fire-model', type=str, required=True,
                       help='Path to YOLO fire detection model')
    parser.add_argument('--material-model', type=str, default=None,
                       help='Path to material classification model')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='Base confidence threshold (default: 0.15)')
    
    args = parser.parse_args()
    
    # Initialize
    system = EnhancedFireAnalysis(
        args.fire_model,
        args.material_model,
        args.conf
    )
    
    # Process
    if args.video:
        system.process_video(args.video, args.output)
    else:
        print("❌ Error: Please specify --video")


if __name__ == '__main__':
    main()