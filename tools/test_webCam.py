import cv2

index = 0  # Try 0, 1, 2, etc.
cap = cv2.VideoCapture(index)
if cap.isOpened():
    print(f"Webcam index {index} is working.")
else:
    print(f"Webcam index {index} is not working.")
cap.release()
