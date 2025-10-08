# Face Recognition & Object Tracking Project

This project uses **DeepFace** for face recognition and **OpenCV** for object tracking (KCF tracker).  
It can detect faces in images/videos and verify them against a reference image.  

## 🚀 New: FastAPI Web API

The project now includes a **FastAPI web API** that provides REST endpoints for face verification, making it easy to integrate with web applications and other services.

---

## 📦 Requirements

The project has been tested with the following versions:

- numpy>=1.23.5,<1.24
- opencv-contrib-python==4.5.5.64  
- deepface==0.0.93  
- torch==2.0.1  
- torchvision==0.15.2  
- torchaudio==2.0.2  
- tensorflow==2.12.0  
- keras==2.12.0  
- ultralytics==8.3.169  
- fastapi==0.104.1
- uvicorn==0.24.0
- python-multipart==0.0.6

---

## 🔹 Installation

1️⃣ **Clone this repository**
```bash
git clone https://github.com/WasifAliShah/FYP-AI-module
cd FYP-AI-module
```

2️⃣ **Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Options

### Option 1: FastAPI Web API (Recommended)

**Start the FastAPI server:**
```bash
python run_fastapi.py
```

**Or manually:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Access the API:**
- API Base URL: http://localhost:8000
- Interactive Documentation: http://localhost:8000/docs
- Alternative Documentation: http://localhost:8000/redoc

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /verify-image` - Verify a single image
- `POST /process-video` - Process a video file
- `GET /process-video-stream` - Stream video processing

**Test the API:**
```bash
python test_api.py
```

### Option 2: Standalone Script

**Run the original script:**
```bash
python 1st.py
```

⚠ **Note:**  
The `1st.py` file contains **placeholders** for image and video paths.  
Replace them with your own file paths before running the script.

---

## 📡 API Usage Examples

### Verify an Image
```bash
curl -X POST "http://localhost:8000/verify-image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     -F "threshold=0.5"
```

### Process a Video
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_path": "combine.mp4",
       "reference_image_path": "wasif7.jpg",
       "threshold": 0.5,
       "cooldown": 5,
       "skip_frames": 10
     }'
```

### Stream Video Processing
```bash
curl "http://localhost:8000/process-video-stream?video_path=combine.mp4"
```

---

## 🔧 Configuration

The FastAPI application automatically loads:
- YOLOv8 face detection model: `yolov8n-face-lindevs.pt`
- Reference image: `wasif7.jpg`

Make sure these files are present in the project directory.

---

## 📝 Important Notes

- The FastAPI version maintains all the same functionality as the original script
- All dependencies and versions remain exactly the same
- The API provides both synchronous and streaming endpoints
- Face verification uses the same ArcFace model and cosine similarity threshold
- Video processing includes the same tracking and detection logic

