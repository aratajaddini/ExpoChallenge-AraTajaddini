from ultralytics import YOLO


model = YOLO("yolo11n.pt")


results = model.train(
    data="/content/taco_dataset/data.yaml",
    epochs=100,             
    imgsz=640,                
    batch=16,                
    device=0,                 
    project="TACO_YOLOv11",
    name="custom_18class_run",
    save=True,
    patience=15,              


    # Data Augmentation
    mosaic=1.0,              
    mixup=0.15,             
    copy_paste=0.1,           
    degrees=15.0,           
    translate=0.1,            
    scale=0.5,               
    shear=2.0,                
    perspective=0.0005,       
    fliplr=0.5,              
    flipud=0.2,              
    hsv_h=0.015,              
    hsv_s=0.7,                
    hsv_v=0.4,             
    erasing=0.4,              
)

print("Training completed successfully")