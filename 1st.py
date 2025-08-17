# Non detection --- Processing completed in 35.92 seconds ---
# Face was NOT detected in the video.
# (storefront310) PS C:\cameras2\storefront310> python fifth.py

# 0: 384x640 1 face, 248.1ms
# Speed: 12.3ms preprocess, 248.1ms inference, 14.4ms postprocess per image at shape (1, 3, 384, 640)
# Face verified at frame 5

# --- Processing completed in 46.01 seconds ---
# Face was detected and verified in the video.


# import cv2
# import torch
# import time
# from deepface import DeepFace
# from ultralytics import YOLO
# from scipy.spatial.distance import cosine

# # Load YOLOv8 face detection model
# model = YOLO("yolov8n-face-lindevs.pt")

# # Load reference image and calculate its embedding
# ref_img_path = "wasif.jpg"
# ref_img = cv2.imread(ref_img_path)
# ref_embedding = DeepFace.represent(ref_img, model_name="ArcFace", detector_backend='retinaface', enforce_detection=False)[0]["embedding"]

# # Function to compare embeddings
# def is_match(embedding1, embedding2, threshold=0.5):
#     return cosine(embedding1, embedding2) < threshold

# # Initialize video
# cap = cv2.VideoCapture("video5.mp4")
# tracker = None
# tracking = False
# frame_count = 0
# skip_frames = 5
# cooldown = 5  # seconds
# last_match_time = 0
# verified_once = False

# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = frame.copy()

#     if tracking:
#         success, bbox = tracker.update(frame)
#         if success:
#             x, y, w, h = map(int, bbox)
#             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(display_frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             tracking = False  # lost the face, try detecting again
#     else:
#         if frame_count % skip_frames == 0:
#             results = model.predict(frame, conf=0.5)
#             boxes = results[0].boxes

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 face_crop = frame[y1:y2, x1:x2]

#                 # Skip if recently matched
#                 if time.time() - last_match_time < cooldown:
#                     continue

#                 # Get embedding of current face
#                 try:
#                     embedding = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
#                 except:
#                     continue  # skip if face representation fails

#                 if is_match(ref_embedding, embedding):
#                     print("Face verified at frame", frame_count)
#                     verified_once = True
#                     last_match_time = time.time()
#                     tracker = cv2.TrackerKCF_create()
#                     tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
#                     tracking = True
#                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(display_frame, "Verified", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     break
#                 else:
#                     # Draw red box for detected but unverified face
#                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(display_frame, "Unverified", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     cv2.imshow("Hybrid Face Verification", display_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"\n--- Processing completed in {elapsed_time:.2f} seconds ---")
# print("Face was" + (" detected and verified in the video." if verified_once else " NOT detected in the video."))


# import cv2
# import time
# import torch
# from deepface import DeepFace
# from ultralytics import YOLO
# from scipy.spatial.distance import cosine
# from concurrent.futures import ThreadPoolExecutor

# # Load YOLOv8 face detection model
# model = YOLO("yolov8n-face-lindevs.pt")

# # Load reference image and calculate its embedding once
# ref_img = cv2.imread("wasif5.jpg")
# ref_embedding = DeepFace.represent(ref_img, model_name="ArcFace", detector_backend='retinaface', enforce_detection=False)[0]["embedding"]

# # Matching logic
# def is_match(embedding1, embedding2, threshold=0.5):
#     return cosine(embedding1, embedding2) < threshold

# # Global state
# cap = cv2.VideoCapture("combine.mp4")
# frame_count = 0
# skip_frames = 5
# cooldown = 5  # seconds between matches
# last_match_time = 0
# verified_once = False
# tracking = False
# tracker = None
# executor = ThreadPoolExecutor(max_workers=2)

# def process_face(face_img, frame_idx, bbox, frame):
#     global last_match_time, verified_once, tracking, tracker

#     try:
#         embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
#         if is_match(ref_embedding, embedding):
#             print(f"[✅ Verified] at frame {frame_idx}")
#             verified_once = True
#             last_match_time = time.time()
#             x1, y1, x2, y2 = bbox
#             tracker = cv2.TrackerKCF_create()
#             tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
#             tracking = True
#     except Exception as e:
#         print(f"[!] Face processing error: {e}")

# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = frame.copy()

#     # TRACKING
#     if tracking:
#         # Ensure frame has 3 channels (BGR)
#         if len(frame.shape) == 2 or frame.shape[2] == 1:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         success, bbox = tracker.update(frame)
#         if success:
#             x, y, w, h = map(int, bbox)
#             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(display_frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         else:
#             tracking = False  # lost face
#     elif frame_count % skip_frames == 0:
#         results = model.predict(frame, conf=0.5, verbose=False)
#         boxes = results[0].boxes

#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_crop = frame[y1:y2, x1:x2]

#             if time.time() - last_match_time < cooldown:
#                 continue

#             # Launch async embedding + match
#             executor.submit(process_face, face_crop, frame_count, (x1, y1, x2, y2), frame)

#             # Draw red box while waiting for verification
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(display_frame, "Verifying...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     cv2.imshow("Threaded Face Verification", display_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# executor.shutdown(wait=True)

# end_time = time.time()
# print(f"\n🕒 Completed in {end_time - start_time:.2f} seconds")
# print("🎯 Face was" + (" detected and verified." if verified_once else " NOT detected."))



# v3 minor changes from the previous one:
# import cv2
# import time
# import torch
# from deepface import DeepFace
# from ultralytics import YOLO
# from scipy.spatial.distance import cosine
# from concurrent.futures import ThreadPoolExecutor

# # Load YOLOv8 face detection model
# model = YOLO("yolov8n-face-lindevs.pt")

# # Load reference image and calculate its embedding once
# ref_img = cv2.imread("wasif7.jpg")
# ref_embedding = DeepFace.represent(ref_img, model_name="ArcFace", detector_backend='retinaface', enforce_detection=False)[0]["embedding"]

# # Matching logic
# def is_match(embedding1, embedding2, threshold=0.5):
#     return cosine(embedding1, embedding2) < threshold

# # Global state
# cap = cv2.VideoCapture("video5.mp4")
# frame_count = 0
# skip_frames = 5
# cooldown = 5  # seconds between matches
# last_match_time = 0
# verified_once = False
# tracking = False
# tracker = None
# executor = ThreadPoolExecutor(max_workers=2)

# def process_face(face_img, frame_idx, bbox, frame):
#     global last_match_time, verified_once, tracking, tracker

#     try:
#         embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
#         if is_match(ref_embedding, embedding):
#             print(f"[✅ Verified] at frame {frame_idx}")
#             verified_once = True
#             last_match_time = time.time()
#             x1, y1, x2, y2 = bbox
#             tracker = cv2.TrackerKCF_create()  # More robust than KCF
#             tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
#             tracking = True
#     except Exception as e:
#         print(f"[!] Face processing error: {e}")

# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = frame.copy()

#     # TRACKING
#     if tracking:
#         if len(frame.shape) == 2 or frame.shape[2] == 1:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         success, bbox = tracker.update(frame)
#         if success:
#             x, y, w, h = map(int, bbox)
#             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(display_frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         else:
#             tracking = False
#     elif frame_count % skip_frames == 0:
#         results = model.predict(frame, conf=0.5, verbose=False)
#         boxes = results[0].boxes

#         for box in boxes:
#             bbox = box.xyxy[0].cpu().numpy().astype(int)
#             x1, y1, x2, y2 = bbox

#             # Ensure bounding box stays within frame bounds
#             h, w, _ = frame.shape
#             x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

#             face_crop = frame[y1:y2, x1:x2]

#             if time.time() - last_match_time < cooldown:
#                 continue

#             # Launch async embedding + match (pass copies)
#             executor.submit(
#                 process_face,
#                 face_crop.copy(),
#                 frame_count,
#                 (x1, y1, x2, y2),
#                 frame.copy()
#             )

#             # Draw red box while waiting for verification
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(display_frame, "Verifying...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     cv2.imshow("Threaded Face Verification", display_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# executor.shutdown(wait=True)

# end_time = time.time()
# print(f"\n🕒 Completed in {end_time - start_time:.2f} seconds")
# print("🎯 Face was" + (" detected and verified." if verified_once else " NOT detected."))


import numpy as np 
import cv2
import time
import torch
from deepface import DeepFace
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor

# Load YOLOv8 face detection model
model = YOLO("yolov8n-face-lindevs.pt")

# Load reference image and calculate its embedding once
ref_img = cv2.imread("wasif7.jpg")
ref_embedding = DeepFace.represent(
    ref_img, model_name="ArcFace", detector_backend='retinaface', enforce_detection=False
)[0]["embedding"]

# Matching logic
def is_match(embedding1, embedding2, threshold=0.5):
    return cosine(embedding1, embedding2) < threshold

# Global state
cap = cv2.VideoCapture("combine.mp4")
frame_count = 0

# --- FPS-based skip_frames ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:  # fallback
    fps = 30
skip_frames = max(1, round(fps / 3))  # ~3 detections per second
print(f"[🎬 Init] Video FPS={fps:.1f}, using skip_frames={skip_frames}")

cooldown = 5  # seconds between matches
last_match_time = 0
verified_once = False
tracking = False
tracker = None
executor = ThreadPoolExecutor(max_workers=2)

def process_face(face_img, frame_idx, bbox, frame):
    global last_match_time, verified_once, tracking, tracker

    try:
        embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        if is_match(ref_embedding, embedding):
            print(f"[✅ Verified] at frame {frame_idx}")
            verified_once = True
            last_match_time = time.time()
            x1, y1, x2, y2 = bbox
            tracker = cv2.TrackerCSRT_create()  # More robust than KCF
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            tracking = True
    except Exception as e:
        print(f"[!] Face processing error: {e}")

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    # TRACKING
    if tracking:
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if frame is None or frame.shape[2] != 3:
            tracking = False
            continue

        try:    
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("[!] Tracker lost target, re-detecting...")
                tracking = False
        except cv2.error as e:
            print(f"[!] Tracker error: {e}")
            tracking = False

    # FALLBACK TO DETECTION IF NOT TRACKING
    if not tracking and frame_count % skip_frames == 0:
        results = model.predict(frame, conf=0.5, imgsz=320, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox

            # Ensure bounding box stays within frame bounds
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]

            if time.time() - last_match_time < cooldown:
                continue

            # Launch async embedding + match (pass copies)
            executor.submit(
                process_face,
                face_crop.copy(),
                frame_count,
                (x1, y1, x2, y2),
                frame.copy()
            )

            # Draw red box while waiting for verification
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, "Verifying...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Threaded Face Verification", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)

end_time = time.time()
print(f"\n🕒 Completed in {end_time - start_time:.2f} seconds")
print("🎯 Face was" + (" detected and verified." if verified_once else " NOT detected."))
