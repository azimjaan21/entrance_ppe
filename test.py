
from ultralytics import YOLO
import os

model = YOLO('######################')


image_path = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\21'
output_folder = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\ppe\YOLO Pred'
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
