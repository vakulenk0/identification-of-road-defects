import os
import logging
from datetime import datetime

import torch
from ultralytics import YOLO
import cv2 

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_and_log(message):
    print(message)
    logging.info(message)

cuda_available = torch.cuda.is_available()
print_and_log(f"CUDA доступен: {cuda_available}")


if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    print_and_log(f"Используется устройство: {device_name}")
else:
    print_and_log("Обучение будет производиться на CPU")
device = torch.device("cuda" if cuda_available else "cpu")


model_path = "C:/Repositories/diplom/yolo11s.pt"
print_and_log(f"Загрузка модели из: {model_path}")
model = YOLO(model_path).to(device=device)


data_path = "./datasets/data.yaml"


epochs = 75
batch_size = 16
image_size = 640
workers = 4
experiment_name = "yolo11m-road-damage"
project_dir = "runs/train"
early_stop_patience = 20
print_and_log("Запуск обучения модели YOLOv11...")
print_and_log(f"Эпохи: {epochs}, Batch size: {batch_size}, Размер изображения: {image_size}")


results = model.train(
    data=data_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=image_size,
    device=0 if cuda_available else 'cpu',
    workers=workers,
    name=experiment_name,
    patience=early_stop_patience,
    project=project_dir
)

print_and_log("Обучение завершено.")


best_model_path = os.path.join(project_dir, experiment_name, "weights", "best.pt")
if os.path.exists(best_model_path):
    print_and_log(f"Лучшее состояние модели сохранено: {best_model_path}")
else:
    print_and_log("Файл с лучшим состоянием модели не найден.")


if results:
    try:
        metrics = results.results_dict
        for key, value in metrics.items():
            print_and_log(f"{key}: {value}")
    except Exception as e:
        print_and_log(f"Ошибка при извлечении метрик: {e}")

