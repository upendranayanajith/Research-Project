import cv2
import sys
import os

url = "rtsp://172.29.230.158:554/stream1"
if len(sys.argv) > 1:
    url = sys.argv[1]

print(f"Testing RTSP stream: {url}")

# Try default
print("Attempt 1: Default VideoCapture")
cap = cv2.VideoCapture(url)
if cap.isOpened():
    print("SUCCESS with default!")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read frame of shape {frame.shape}")
    cap.release()
else:
    print("FAILED with default.")

# Try with TCP
print("\nAttempt 2: Forcing TCP transport")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if cap.isOpened():
    print("SUCCESS with TCP!")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read frame of shape {frame.shape}")
    cap.release()
else:
    print("FAILED with TCP.")

# Try with UDP
print("\nAttempt 3: Forcing UDP transport")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if cap.isOpened():
    print("SUCCESS with UDP!")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read frame of shape {frame.shape}")
    cap.release()
else:
    print("FAILED with UDP.")
