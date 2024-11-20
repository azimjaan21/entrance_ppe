from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8m.pt') 

    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='cuda',  
        lr0=0.01,  #learning rate
        weight_decay=0.0005,  # Regularization
        patience=10,  # Early stopping
        optimizer='SGD',
        momentum=0.937,
        augment=True,  
        plots=True,  
        verbose=True  
    )

if __name__ == '__main__':
    train_model()

