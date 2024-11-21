
from ultralytics import YOLO
import os

model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\ppe_entrance.pt')


image_path = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\predict_samples\1.jpg'
output_folder = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\YOLO Pred'
os.makedirs(output_folder, exist_ok=True)

# Run model prediction
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,       
    save_txt=True,   
    project=output_folder, 
    device="cuda"
)
