from ultralytics import YOLO
import torch

def validate_model():
    # Load the trained model
    model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\ppe_entrance.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running validation on device: {device}")

    results = model.val(
       data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\entrance_ppe\data.yaml',
        batch=16,       
        imgsz=640,       
        device='cuda',   
        conf=0.25,      
        iou=0.5,         
        verbose=True     
    )

    # Print and analyze results
    print("\nValidation Results:")
    print(results)  # Shows overall metrics (mAP, precision, recall)

    # Save results to a file for further analysis
    with open("validation_results.txt", "w") as file:
        file.write(str(results))

if __name__ == '__main__':
    validate_model()

