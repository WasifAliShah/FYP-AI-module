import numpy as np
import cv2
import time
import torch
from deepface import DeepFace
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64
import io
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification API",
    description="AI-powered face detection and verification system using YOLOv8 and DeepFace",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and state
model = None
ref_embedding = None
tracking_state = {
    "tracking": False,
    "tracker": None,
    "verified_once": False,
    "last_match_time": 0,
    "executor": None
}

# Pydantic models for request/response
class VerificationRequest(BaseModel):
    threshold: float = 0.5
    cooldown: int = 5

class VerificationResponse(BaseModel):
    verified: bool
    confidence: float
    frame_count: int
    processing_time: float
    message: str

class VideoProcessingRequest(BaseModel):
    video_path: str
    reference_image_path: str
    threshold: float = 0.5
    cooldown: int = 5
    skip_frames: int = 10

class VideoProcessingResponse(BaseModel):
    verified: bool
    total_frames: int
    processing_time: float
    message: str

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and reference embedding on startup"""
    global model, ref_embedding, tracking_state
    
    try:
        logger.info("Loading YOLOv8 face detection model...")
        model = YOLO("yolov8n-face-lindevs.pt")
        
        logger.info("Loading reference image and calculating embedding...")
        ref_img = cv2.imread("wasif7.jpg")
        if ref_img is None:
            raise ValueError("Reference image 'wasif7.jpg' not found")
        
        ref_embedding = DeepFace.represent(
            ref_img, 
            model_name="ArcFace", 
            detector_backend='retinaface', 
            enforce_detection=False
        )[0]["embedding"]
        
        # Initialize thread pool executor
        tracking_state["executor"] = ThreadPoolExecutor(max_workers=2)
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if tracking_state["executor"]:
        tracking_state["executor"].shutdown(wait=True)
    logger.info("Application shutdown complete")

# Utility functions
def is_match(embedding1, embedding2, threshold=0.5):
    """Check if two face embeddings match"""
    return cosine(embedding1, embedding2) < threshold

def process_face_async(face_img, frame_idx, bbox, frame, threshold=0.5):
    """Process face verification asynchronously"""
    global ref_embedding, tracking_state
    
    try:
        embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        confidence = 1 - cosine(ref_embedding, embedding)
        
        if is_match(ref_embedding, embedding, threshold):
            logger.info(f"[✅ Verified] at frame {frame_idx} with confidence {confidence:.3f}")
            tracking_state["verified_once"] = True
            tracking_state["last_match_time"] = time.time()
            
            x1, y1, x2, y2 = bbox
            tracking_state["tracker"] = cv2.TrackerCSRT_create()
            tracking_state["tracker"].init(frame, (x1, y1, x2 - x1, y2 - y1))
            tracking_state["tracking"] = True
            
            return True, confidence
        else:
            return False, confidence
            
    except Exception as e:
        logger.error(f"[!] Face processing error: {e}")
        return False, 0.0

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Face Verification API",
        "version": "1.0.0",
        "endpoints": {
            "verify_image": "/verify-image",
            "process_video": "/process-video",
            "process_video_stream": "/process-video-stream",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "reference_loaded": ref_embedding is not None
    }

@app.post("/verify-image", response_model=VerificationResponse)
async def verify_image(
    file: UploadFile = File(...),
    request: VerificationRequest = VerificationRequest()
):
    """Verify a single image against the reference face"""
    if model is None or ref_embedding is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    start_time = time.time()
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect faces
        results = model.predict(img, conf=0.5, imgsz=320, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return VerificationResponse(
                verified=False,
                confidence=0.0,
                frame_count=0,
                processing_time=time.time() - start_time,
                message="No faces detected in image"
            )
        
        # Process each detected face
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure bounding box stays within frame bounds
            h, w, _ = img.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            face_crop = img[y1:y2, x1:x2]
            
            # Verify face
            verified, confidence = process_face_async(
                face_crop, 0, (x1, y1, x2, y2), img, request.threshold
            )
            
            if verified:
                return VerificationResponse(
                    verified=True,
                    confidence=confidence,
                    frame_count=1,
                    processing_time=time.time() - start_time,
                    message="Face verified successfully"
                )
        
        return VerificationResponse(
            verified=False,
            confidence=0.0,
            frame_count=1,
            processing_time=time.time() - start_time,
            message="No matching face found"
        )
        
    except Exception as e:
        logger.error(f"Error in verify_image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-video", response_model=VideoProcessingResponse)
async def process_video(request: VideoProcessingRequest):
    """Process a video file for face verification"""
    if model is None or ref_embedding is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    start_time = time.time()
    
    try:
        # Load video
        cap = cv2.VideoCapture(request.video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        skip_frames = max(1, round(fps / 3)) if request.skip_frames == 10 else request.skip_frames
        
        logger.info(f"Processing video: FPS={fps:.1f}, skip_frames={skip_frames}")
        
        # Reset tracking state
        tracking_state["tracking"] = False
        tracking_state["tracker"] = None
        tracking_state["verified_once"] = False
        tracking_state["last_match_time"] = 0
        
        frame_count = 0
        verified = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Tracking logic
            if tracking_state["tracking"]:
                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                if frame is None or frame.shape[2] != 3:
                    tracking_state["tracking"] = False
                    continue
                
                try:
                    success, bbox = tracking_state["tracker"].update(frame)
                    if success:
                        # Face is being tracked
                        verified = True
                    else:
                        logger.info("[!] Tracker lost target, re-detecting...")
                        tracking_state["tracking"] = False
                except cv2.error as e:
                    logger.error(f"[!] Tracker error: {e}")
                    tracking_state["tracking"] = False
            
            # Detection logic
            if not tracking_state["tracking"] and frame_count % skip_frames == 0:
                results = model.predict(frame, conf=0.5, imgsz=320, verbose=False)
                boxes = results[0].boxes
                
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Ensure bounding box stays within frame bounds
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if time.time() - tracking_state["last_match_time"] < request.cooldown:
                        continue
                    
                    # Process face verification
                    face_verified, confidence = process_face_async(
                        face_crop, frame_count, (x1, y1, x2, y2), frame, request.threshold
                    )
                    
                    if face_verified:
                        verified = True
                        break
        
        cap.release()
        processing_time = time.time() - start_time
        
        return VideoProcessingResponse(
            verified=verified,
            total_frames=frame_count,
            processing_time=processing_time,
            message="Face was detected and verified." if verified else "Face was NOT detected."
        )
        
    except Exception as e:
        logger.error(f"Error in process_video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/process-video-stream")
async def process_video_stream(video_path: str):
    """Stream video processing with real-time face verification"""
    if model is None or ref_embedding is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    def generate_frames():
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield f"data: {JSONResponse(content={'error': 'Could not open video file'})}\n\n"
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            skip_frames = max(1, round(fps / 3))
            
            frame_count = 0
            tracking_state["tracking"] = False
            tracking_state["tracker"] = None
            tracking_state["verified_once"] = False
            tracking_state["last_match_time"] = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Tracking logic
                if tracking_state["tracking"]:
                    if len(frame.shape) == 2 or frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    if frame is None or frame.shape[2] != 3:
                        tracking_state["tracking"] = False
                        continue
                    
                    try:
                        success, bbox = tracking_state["tracker"].update(frame)
                        if success:
                            x, y, w, h = map(int, bbox)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Tracking", (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            tracking_state["tracking"] = False
                    except cv2.error as e:
                        tracking_state["tracking"] = False
                
                # Detection logic
                if not tracking_state["tracking"] and frame_count % skip_frames == 0:
                    results = model.predict(frame, conf=0.5, imgsz=320, verbose=False)
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        h, w, _ = frame.shape
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        
                        face_crop = frame[y1:y2, x1:x2]
                        
                        if time.time() - tracking_state["last_match_time"] < 5:
                            continue
                        
                        # Process face asynchronously
                        tracking_state["executor"].submit(
                            process_face_async,
                            face_crop.copy(),
                            frame_count,
                            (x1, y1, x2, y2),
                            frame.copy()
                        )
                        
                        # Draw red box while waiting for verification
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Verifying...", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', display_frame)
                frame_bytes = buffer.tobytes()
                frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                
                # Send frame data
                yield f"data: {JSONResponse(content={'frame': frame_b64, 'frame_count': frame_count, 'verified': tracking_state['verified_once']})}\n\n"
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            yield f"data: {JSONResponse(content={'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_frames(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
