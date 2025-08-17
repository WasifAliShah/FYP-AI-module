# Face Recognition & Object Tracking Project

This project uses **DeepFace** for face recognition and **OpenCV** for object tracking (KCF tracker).  
It can detect faces in images/videos and verify them against a reference image.  

⚠ **Note:**  
The `1st.py` file contains **placeholders** for image and video paths.  
Replace them with your own file paths before running the script.

---

## 📦 Requirements

The project has been tested with the following versions:

- numpy==1.26.4  
- opencv-contrib-python==4.5.5.64  
- deepface==0.0.79  
- torch==2.0.1  
- torchvision==0.15.2  
- torchaudio==2.0.2  
- tensorflow==2.12.0  
- keras==2.12.0  
- ultralytics==8.3.169  

---

## 🔹 Installation

1️⃣ **Clone this repository**
```bash
git clone https://github.com/WasifAliShah/FYP-AI-module
cd FYP-AI-module
```
2️⃣ **Create a virtual environment (recommended)**
```bash
python -m venv deepface_env
deepface_env\Scripts\activate   # On Windows
source deepface_env/bin/activate  # On Mac/Linux
```
3️⃣ **Install dependencies**
```bash
pip install numpy==1.26.4
pip install opencv-contrib-python==4.5.5.64
pip install deepface==0.0.79
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install tensorflow==2.12.0 keras==2.12.0
pip install ultralytics==8.3.169
```

▶ **Usage**
Run the script:
```bash
python 1st.py
```

## Important:

Replace the placeholder paths in 1st.py for the image and video with your own files.
Ensure that your files exist in the given paths before running.

