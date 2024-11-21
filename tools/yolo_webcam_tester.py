import cv2
from ultralytics import YOLO

 #my custom YOLOv8
model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\ppe_entrance.pt") 

#webcam
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # predictions on the frame
    results = model(source=frame, show=True, conf=0.5)

    for r in results:
        boxes = r.boxes
        print(boxes)  

    # (OpenCV window)
    cv2.imshow('YOLOv8 Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
