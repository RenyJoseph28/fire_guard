# 🔥 Fire Guard: Advanced Fire & Smoke Detection System

Fire Guard is an intelligent, real-time fire and smoke detection system built with Django and YOLOv8. It goes beyond simple fire detection by analyzing the surrounding context, identifying flammable materials (like gas cylinders), and classifying the type of fire to provide critical, actionable intelligence to first responders.

## 🌟 Key Features

*   **Real-Time Detection:** Analyze live CCTV/camera feeds or upload pre-recorded videos for instant fire and smoke detection.
*   **Dual-Model AI Architecture:**
    *   **Fire/Smoke Model:** Specifically trained to detect flames and smoke using Ultralytics YOLOv8.
    *   **Material Context Model:** A secondary YOLOv8 model trained to detect contextual objects like gas cylinders, vehicles, electronics, and furniture.
*   **Smart False-Positive Filtering:** Advanced heuristic filters to differentiate real fire from sunlight/glare, and real smoke from fog, clouds, or blue skies.
*   **Fire Classification (A, B, C, K):** Automatically classifies the fire type based on nearby burning materials.
    *   *Example:* Detecting a fire near a `laptop` classifies it as Class C (Electrical). Detecting near an `oven` flags Class K (Kitchen/Grease).
*   **⚠️ Explosion Risk Warnings:** Specifically identifies highly flammable contexts (e.g., Gas Cylinders, Flammable liquids) and escalates the alert to **CRITICAL**.
*   **Multi-Channel Push Alerts:**
    *   **Email Notifications:** Sends detailed snapshots, severity levels, and recommended extinguisher types to registered emergency contacts.
    *   **Firebase Cloud Messaging (FCM):** Sends immediate push notifications to mobile devices.
*   **Comprehensive Admin Dashboard:** Manage alert recipients, view past alerts with snapshots, upload footage, and review detection confidence metrics.

---

## 🛠️ Technology Stack

*   **Backend:** Python 3.x, Django 5.x
*   **Computer Vision / AI:** Ultralytics YOLOv8, OpenCV (cv2), NumPy
*   **Database:** SQLite (default) / MySQL (supported)
*   **Notifications:** Firebase Admin SDK (FCM for Push), Django Core Mail (SMTP)
*   **Image Processing:** Pillow (PIL)

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fire_guard.git
cd fire_guard
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the YOLO Models

Ensure the custom YOLO weights are placed in the root directory. The system looks for:
*   `best.pt` (Fire and Smoke detection)
*   `best_material.pt` (Flammable and general materials detection)
*(If missing, the system will fallback to `yolov8n.pt` for basic testing, but performance will be severely degraded).*

### 5. Configure Firebase (For Push Notifications)

Ensure you have your Firebase Admin Service Account JSON file in the root directory. 
By default, the project expects a file similar to `fireguard-fbe22-efa549248415.json` (Update the path in `settings.py` if your file name differs).

### 6. Database Setup

Run the Django migrations to set up the database tables for Users, Alerts, and FCM Devices.

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Create a Superuser

Create an admin account to access the dashboard.

```bash
python manage.py createsuperuser
```

### 8. Run the Development Server

```bash
python manage.py runserver
```

Open your browser and navigate to `http://127.0.0.1:8000/adminpanel/` to log in to the dashboard.

---

## 💻 How to Use

1.  **Dashboard / Login:** Navigate to `/adminpanel/login/` and log in with your superuser credentials.
2.  **Add Alert Recipients:** Go to **Manage Recipients** to add email addresses that should be notified during an incident. You can toggle whether they receive all alerts or only *CRITICAL* ones.
3.  **Video Upload Analysis:** Go to **Upload Video**, select a `.mp4` file, and let the system process it. It will break down the video, apply the smart filters, and generate alerts for any detected emergencies.
4.  **Live Detection:** Go to **Live CameraFeed** to activate your webcam or connected IP camera. The YOLOv8 models will analyze the feed frame-by-frame in real time.

---

## 🧠 How the Smart Filter Works

Fire Guard uses a sophisticated combination of AI and traditional Computer Vision (OpenCV) algorithms to prevent false alarms:
*   **Fire vs. Sunlight:** Analyzes the RGB ratios and texture turbulence of bounding boxes. Smooth, bright white/yellow regions with low turbulence are heavily penalized to prevent sun glare from triggering a fire alert.
*   **Smoke vs. Fog/Clouds:** Checks region brightness, saturation, size, and smoothness. It also checks the upper portion of the frame for blue skies. Dark, textured smoke is allowed, while smooth, wide, unsaturated fog is rejected unless confidence is extremely high.
*   **Red Cylinder Heuristic:** While the material YOLO model detects cylinders, an OpenCV HSV-color filter acts as a safety-net to specifically hunt for maroon/red cylindrical shapes directly below any detected flames.

---

## 🤝 Contributing

Contributions are welcome! If you want to improve the smart filtering algorithms, add support for more IP Camera protocols, or upgrade the UI, please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
