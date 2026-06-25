# 🔥 Fire Guard — Team Work Division (4 Students)

Each student works on **one part from each layer** of the project,
so everyone can explain the full system during the review.

---

## 👩‍💻 Student 1

### What you work on:
- **AI:** The filter that blocks fake fire alerts (e.g. sunlight being mistaken for fire, fog being mistaken for smoke)
- **Backend:** The database — design the tables that store Alerts, Recipients, and Devices
- **Frontend:** The main Dashboard page — shows alert history, severity, and snapshots
- **Notifications:** Email alerts — make sure emails are sent with the fire photo, fire class, and extinguisher tip

### Your key files:
```
adminpanel/detect_utils.py     → (filter section only)
adminpanel/models.py           → Alert, Recipient, Device models
adminpanel/migrations/         → Run and manage DB migrations
templates/adminpanel/dashboard.html
fire_guard/settings.py         → (EMAIL / SMTP section)
```

---

## 👨‍💻 Student 2

### What you work on:
- **AI:** Fire classification — decides if the fire is Class A, B, C, or K based on what materials are nearby
- **Backend:** Video upload — receives the uploaded video, runs detection, saves the alert
- **Frontend:** Video upload page — shows progress bar and the detection results with colored bounding boxes
- **Notifications:** Firebase push notifications — sends instant alerts to mobile devices

### Your key files:
```
adminpanel/detect_utils.py     → (fire class A/B/C/K section)
adminpanel/views.py            → (upload_video view)
templates/adminpanel/upload_video.html
static/firebase-messaging-sw.js
static/js/firebase-messaging.js
```

---

## 👩‍💻 Student 3

### What you work on:
- **AI:** Frame processing — how the video is broken into frames and sent to the YOLO model
- **Backend:** Live camera detection + Login/Logout system
- **Frontend:** Live detection page — shows the live camera stream with real-time bounding boxes
- **Notifications:** PWA setup — makes the app installable and work offline

### Your key files:
```
run_detection.py               → Frame sampling pipeline
enhanced_fire_analysis.py      → Extra analysis logic
adminpanel/views.py            → (live_detection view)
user/views.py + user/urls.py   → Login / Logout
templates/adminpanel/live_detection.html
static/service-worker.js
static/manifest.json
```

---

## 👨‍💻 Student 4

### What you work on:
- **AI:** Gas cylinder detection — uses color detection (OpenCV) to find red/maroon cylinders near fire
- **Backend:** URL routing and project settings + managing alert recipients (add/delete/toggle)
- **Frontend:** Login page + Recipients page (the list of people who get notified)
- **Notifications:** Security check (make sure no passwords or keys are uploaded to GitHub) + write the deployment guide

### Your key files:
```
adminpanel/detect_utils.py     → (HSV cylinder heuristic section)
adminpanel/detect_utils_patch.py
fire_guard/settings.py         → (general settings)
fire_guard/urls.py + adminpanel/urls.py
adminpanel/views.py            → (recipients add/delete views)
templates/adminpanel/login.html
templates/adminpanel/recipients_list.html
templates/adminpanel/add_recipient.html
.gitignore                     → Security check
DEPLOYMENT.md                  → Write this guide
```

---

## 🔗 How the system flows (simple version)

```
Video / Camera
     │
     ▼
Student 3 → breaks video into frames, feeds to YOLO model
     │
     ▼
Student 1 → filters out fake detections (no false alarms)
     │
     ▼
Student 2 → classifies the fire type (A/B/C/K), saves the alert
     │
     ├──► Student 1 → stores alert in database, sends EMAIL
     └──► Student 2 → sends PUSH NOTIFICATION to phone
                │
                ▼
         Student 1 → shows alert on DASHBOARD
```

---

## ✅ Quick Summary Table

| Student | AI Task | Backend Task | Frontend Page | Notification |
|---------|---------|-------------|---------------|--------------|
| **1** | False-positive filter | DB models & migrations | Dashboard | Email (SMTP) |
| **2** | Fire classification (A/B/C/K) | Video upload view | Upload video page | FCM Push |
| **3** | Frame pipeline (YOLO input) | Live detection + Login | Live feed page | PWA / Offline |
| **4** | Gas cylinder detection (HSV) | URL routing + Recipients | Login + Recipients | Security + Deployment |

---

> 💡 **Tip for review:** Each student should be ready to explain:
> 1. What their AI/logic part does
> 2. How the backend connects it
> 3. What the user sees on screen
> 4. How the alert/notification reaches the user

---

---

# 🧠 Model Making & Training Phase — Work Division

This covers everything that happened **before** the app was built —
how the YOLO models (`best.pt` and `best_material.pt`) were created.

---

## 📋 Stages of Model Making (in order)

```
1. Data Collection
       ↓
2. Data Annotation (Labelling)
       ↓
3. Data Preprocessing
       ↓
4. Model Training
       ↓
5. Model Evaluation & Testing
       ↓
6. Model Export & Integration
```

---

## 👩‍💻 Student 1 — Data Collection

### What you do:
Gather all the raw images and videos needed to train the model.

### Tasks:
- [ ] Collect **fire and smoke images/videos** from the internet (Google, Kaggle, YouTube)
- [ ] Collect **non-fire images** (sunlight, candles, lamps) — needed to teach the model what is NOT fire
- [ ] Collect **smoke look-alike images** (fog, clouds, steam, dust) — so the model learns what is NOT smoke
- [ ] Collect images of **flammable materials** (gas cylinders, vehicles, electronics, furniture)
- [ ] Organize all images into folders:
  ```
  dataset/
    fire/
    smoke/
    no_fire/
    materials/
  ```
- [ ] Make sure dataset has **enough variety** — different lighting, angles, distances, indoors/outdoors

### Tools used:
- Google Images, Kaggle Datasets, Roboflow Universe, YouTube frame grabs
- OpenCV to extract frames from video clips

---

## 👨‍💻 Student 2 — Data Annotation (Labelling)

### What you do:
Draw bounding boxes around each object in every image so the model knows what to learn.

### Tasks:
- [ ] Use **Roboflow** or **LabelImg** tool to annotate images
- [ ] Draw bounding boxes and assign correct class labels:
  - `fire`, `smoke` → for the Fire/Smoke model (`best.pt`)
  - `gas_cylinder`, `vehicle`, `laptop`, `furniture`, `oven` etc. → for the Materials model (`best_material.pt`)
- [ ] Make sure every image has **at least one label**
- [ ] Export annotations in **YOLO format** (`.txt` files with class + coordinates)
- [ ] Split dataset into:
  ```
  train/   → 70% of images (used to learn)
  val/     → 20% of images (used to check during training)
  test/    → 10% of images (used for final testing)
  ```
- [ ] Review labels for mistakes — wrong boxes or wrong class names will ruin training

### Tools used:
- Roboflow (online), LabelImg (desktop), CVAT

---

## 👩‍💻 Student 3 — Data Preprocessing & Model Training

### What you do:
Clean and prepare the data, then run the actual training process.

### Preprocessing tasks:
- [ ] **Resize** all images to `640×640` pixels (YOLOv8 standard input size)
- [ ] **Normalize** pixel values (done automatically by YOLOv8, but verify)
- [ ] Apply **data augmentation** to make the model stronger:
  - Random flipping, rotation, brightness change, cropping
  - Mosaic augmentation (combines 4 images into 1 — built into YOLOv8)
- [ ] Create a `data.yaml` file:
  ```yaml
  path: ./dataset
  train: train/images
  val: val/images
  nc: 2              # number of classes
  names: ['fire', 'smoke']
  ```

### Training tasks:
- [ ] Install Ultralytics YOLOv8: `pip install ultralytics`
- [ ] Train the **Fire/Smoke model**:
  ```
  yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
  ```
- [ ] Train the **Materials model** with its own `data.yaml`
- [ ] Monitor training — watch for loss going down, mAP going up
- [ ] Save the best weights → this becomes `best.pt` and `best_material.pt`

### Tools used:
- Python, Ultralytics YOLOv8, Google Colab (free GPU), or local GPU

---

## 👨‍💻 Student 4 — Model Evaluation & Testing

### What you do:
Test how well the trained model actually works and prepare it for the app.

### Evaluation tasks:
- [ ] Run the model on the **test set** (images the model has never seen):
  ```
  yolo val model=best.pt data=data.yaml
  ```
- [ ] Check key metrics:
  - **Precision** — out of all fire detections, how many were actually fire?
  - **Recall** — out of all actual fires, how many did the model find?
  - **mAP@50** — overall accuracy score (aim for above 80%)
  - **Confusion Matrix** — shows where the model gets confused
- [ ] Test on **real-world videos** (`cylinder.mp4`, `house fire.mp4`, `fog-clouds.mp4`)
- [ ] Check for common failure cases:
  - Is sunlight being detected as fire? → tweak the smart filter (Student 1)
  - Is fog being detected as smoke? → tweak the smart filter (Student 1)
- [ ] **Export model** for use in the Django app:
  - Place `best.pt` and `best_material.pt` in the project root folder
- [ ] Write a short **Model Report**:
  - Dataset size, training time, final mAP score, failure cases found

### Tools used:
- Ultralytics YOLOv8 `val` command, Confusion Matrix plots, sample test videos

---

## ✅ Model Training Phase — Quick Summary Table

| Student | Stage | What happens |
|---------|-------|-------------|
| **1** | Data Collection | Gather fire, smoke, material images from internet & videos |
| **2** | Data Annotation | Draw bounding boxes + labels using Roboflow / LabelImg |
| **3** | Preprocessing + Training | Resize, augment data → run YOLOv8 training → save best.pt |
| **4** | Evaluation + Testing | Check precision/recall/mAP → test on real videos → write report |

---

## 🔁 Full Project Timeline (Both Phases)

```
Phase 1 — Model Making
  Student 1: Collect data
      ↓
  Student 2: Label / annotate data
      ↓
  Student 3: Preprocess + Train model
      ↓
  Student 4: Evaluate + Export model (best.pt)

Phase 2 — App Development
  (Use the TEAM WORK DIVISION section above ↑)
```

> 💡 **Tip:** For the review, each student should be able to explain **both** their model training task AND their app development task. Together, that tells the full story of how Fire Guard was built from scratch.
