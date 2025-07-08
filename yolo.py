import cv2
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11x.pt")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Модель обучается на {device}')
    model.to(device=device)

    
    model.train(
        data="./datasets/data.yaml",     
        epochs=75,                        
        batch=16,                         
        imgsz=640,                        
        workers=4,                        # Количество потоков для загрузки и предобработки данных
        name='yolo11m-road-damage',       
        patience=20,                     
        project='runs/train',                                  
        
        # --- Геометрические аугментации ---
        degrees=10.0,                   # Вращение изображений (угол в градусах)
        translate=0.1,                  # Сдвиг по X/Y (доля от размера)
        scale=0.5,                      # Масштабирование (доля)
        shear=2.0,                      # Сдвиг (угол в градусах)
        flipud=0.5,                     # Вероятность вертикального флипа
        fliplr=0.5,                     # Вероятность горизонтального флипа
        
        # --- Цветовые аугментации ---
        hsv_h=0.015,                    # Изменение оттенка (Hue)
        hsv_s=0.7,                      # Изменение насыщенности (Saturation)
        hsv_v=0.4,                      # Изменение яркости (Value)
        
        # --- Комбинированные аугментации ---
        mosaic=1.0,                     # Mosaic-аугустация (склеивание 4 изображений)
        mixup=0.2,                      # MixUp-аугустация (смешивание изображений)
    )

