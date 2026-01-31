from ultralytics import YOLO
import cv2
import os
import numpy as np

def run_fire_smoke_detection(source, model_path='runs/detect/train/weights/best.pt', conf=0.1):
    """
    Run fire and smoke detection with a Color Balance Filter to reduce false positives from fog/sunlight
    """
    # Load the model
    model = YOLO(model_path)
    
    # Open the video source
    is_webcam = (source == 0 or source == '0')
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    output_dir = 'output/detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'result_with_filter.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video ...")
    print(f"   Source: {source} ({total_frames} frames total)")
    print(f"   Filter: Smart Confidence Shield Active")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model.predict(source=frame, conf=0.05, verbose=False) # Use lower internal conf
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    c = int(box.cls[0])
                    cf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    x1, y1, x2, y2 = xyxy
                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)
                    
                    label = model.names[c]
                    
                    is_valid = True
                    if c == 0: # Fire class
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
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
                            
                            dynamic_thresh = conf + (sunlight_score * 0.5)
                            if cf < dynamic_thresh:
                                is_valid = False

                    else: # Smoke class (or others)
                        # Fog Analysis
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            (mean, std_dev) = cv2.meanStdDev(roi)
                            texture_score = np.mean(std_dev)
                            avg_val = np.mean(mean) # Brightness

                            fog_score = 0.0
                            
                            # 1. Uniformity Check: Fog is very smooth
                            if texture_score < 15: 
                                fog_score += 0.4
                            elif texture_score < 25:
                                fog_score += 0.2
                                
                            # 2. Brightness Check: Fog is often bright white/grey
                            if avg_val > 180:
                                fog_score += 0.2
                                
                            # 3. Size/Shape Check: Fog often covers full width (Horizon)
                            box_w = x2 - x1
                            box_h = y2 - y1
                            if box_w > width * 0.7: # If it's 70% of the screen width
                                fog_score += 0.5

                            # 4. Saturation Check: Fog is very unsaturated (grey/white)
                            b, g, r_v = cv2.mean(roi)[:3]
                            saturation = max(b, g, r_v) - min(b, g, r_v)
                            if saturation < 15: # Very low color (grey)
                                fog_score += 0.3

                            # Max penalty
                            fog_score = min(fog_score, 1.0)
                            
                            # Require MUCH higher confidence if it looks like fog
                            dynamic_thresh = conf + (fog_score * 0.5)
                            
                            if cf < dynamic_thresh:
                                is_valid = False

                    if is_valid:
                        # Draw box
                        color = (0, 0, 255) if c == 0 else (128, 128, 128) 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        text = f"{label} {cf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        text_y = y1 - 10
                        if text_y < text_height:
                            text_y = y1 + text_height + 10
                        
                        cv2.rectangle(frame, (x1, text_y - text_height - 5), (x1 + text_width, text_y + baseline), color, -1)
                        cv2.putText(frame, text, (x1, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Done! Results saved to: {output_path}")

if __name__ == "__main__":
    # ============================================
    # 🎬 VIDEO SOURCE
    # ============================================
    VIDEO_PATH = 'example_1.mp4'  
    
    # CONFIDENCE THRESHOLD
    # Base confidence of 0.15 for real fires.
    # The smart filter will protect against sunlight.
    CONFIDENCE = 0.15
    
    run_fire_smoke_detection(source=VIDEO_PATH, conf=CONFIDENCE)