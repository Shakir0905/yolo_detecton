import os
import cv2
import torch
from ultralytics import YOLO

# Использование GPU, если доступно
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Загрузка модели
model = YOLO("weights/best.pt")
model.to(device)

# Печать классов, которые модель может детектировать
print("Доступные классы модели:")
print(model.names)

# Функция для детекции с веб-камеры
def detect_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.")
        return

    print("Запуск детекции с веб-камеры. Нажмите 'q' для выхода.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        for result in results:
            for box in result.boxes.data:
                class_id = int(box[-1])

                class_name = result.names.get(class_id, "Unknown")  # Получаем имя класса
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Функция для детекции на изображениях из директории
def detect_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, file_name)
            img = cv2.imread(image_path)
            results = model(img, conf=0.5, verbose=False)

            for result in results:
                for box in result.boxes.data:
                    class_id = int(box[-1])
                    class_name = result.names.get(class_id, "Unknown")  # Получаем имя класса
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = box[4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, img)
            print(f"Обработано и сохранено: {output_path}")
# Функция для детекции в видеофайле
def detect_video(video_path, fps_limit=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}.")
        return

    print(f"Запуск детекции в видео. Нажмите 'q' для выхода.")
    delay = int(1000 / fps_limit)  # Задержка между кадрами в миллисекундах

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, verbose=False)
        for result in results:
            for box in result.boxes.data:
                class_id = int(box[-1])
                class_name = result.names.get(class_id, "Unknown")  # Получаем имя класса
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Video Detection", frame)

        # Задержка для уменьшения скорости воспроизведения
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Основной блок для выбора режима
if __name__ == "__main__":
    print("Выберите режим:")
    print("1 - Веб-камера")
    print("2 - Обработка видео")
    print("3 - Обработка изображений из директории")

    choice = input("Введите 1, 2 или 3: ")

    if choice == "1":
        detect_webcam()
    elif choice == "2":
        detect_video("video.mp4")
    elif choice == "3":
        input_dir = "./input_images"   # Путь к входным изображениям
        detect_directory(input_dir, "./output_images")  # Результаты будут демонстрироваться
    else:
        print("Неверный выбор. Завершение программы.")