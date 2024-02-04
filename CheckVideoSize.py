import cv2

video_path = "/mnt/d/Human_Recognition_Pose/Human-Pose-Estimation-Benchmarking-and-Action-Recognition/Human-Falling-Detect-Tracks/Data/Videos/taewondo.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video dimensions: {width} x {height}")

cap.release()
