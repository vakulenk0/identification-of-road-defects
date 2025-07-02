import os
from ultralytics import YOLO
from PIL import Image
import cv2
from collections import Counter

# Замените путь на свой путь к модели
model = YOLO("C:/Repositories/diplom/runs/train/yolo11m-road-damage/weights/best.pt")

def detect_image(image_path, result_folder):
    results = model(image_path)

    # Подсчитываем количество дефектов каждого класса
    defect_counts = Counter()
    for r in results:
        if r.boxes is not None:
            for cls in r.boxes.cls:
                defect_counts[r.names[cls.item()]] += 1

    # Формируем строку с результатами
    result_text = "Обнаружены следующие дефекты:\n"
    for defect_type, count in defect_counts.items():
        result_text += f"- {defect_type}: {count} шт.\n"

    # Создаем папку для результатов, если её нет
    os.makedirs(result_folder, exist_ok=True)

    # Формируем имя файла результата
    result_filename = f"processed_{os.path.basename(image_path)}"
    result_path = os.path.join(result_folder, result_filename)

    # Сохраняем аннотированное изображение
    annotated_img = results[0].plot()
    Image.fromarray(annotated_img).save(result_path)
    
    print("DEBUG detect_image return:", result_text, result_filename)
    return result_text, result_filename

def detect_video(video_path, result_folder):
    results = model(video_path, stream=True)

    os.makedirs(result_folder, exist_ok=True)

    result_filename = f"processed_{os.path.basename(video_path)}"
    result_path = os.path.join(result_folder, result_filename)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (w, h))

    # Подсчитываем количество дефектов каждого класса
    defect_counts = Counter()
    for r in results:
        if r.boxes is not None:
            for cls in r.boxes.cls:
                defect_counts[r.names[cls.item()]] += 1
        frame = r.plot()
        out.write(frame)

    cap.release()
    out.release()

    # Формируем строку с результатами
    result_text = "Обнаружены следующие дефекты:\n"
    for defect_type, count in defect_counts.items():
        result_text += f"- {defect_type}: {count} шт.\n"

    print("DEBUG detect_video return:", result_text, result_filename)
    return result_text, result_filename
