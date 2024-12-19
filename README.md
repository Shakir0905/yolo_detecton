
---

# YOLO Object Detection Project

## Описание

Этот проект использует **YOLO** (You Only Look Once) для детекции 25 кастомных классов, дообученных на данных из **Roboflow**. Модель позволяет выполнять детекцию объектов в изображениях, видео и в реальном времени через веб-камеру.  
Модель была обучена на 12,000 изображениях и достигла **mAP@95** = 86%.

---

## Структура проекта

```plaintext
YOLO/
├── input_images/          # Директория для входных изображений
├── output_images/         # Директория для сохранения обработанных изображений
├── venv/                  # Виртуальное окружение Python
├── weights/               # Директория с весами модели YOLO
│   └── best.pt            # Предобученные веса YOLO
│   └── data.yaml          # Конфигурационный файл данных                           
│   └── image.png          # Процесс обучения модели
├── main.py                # Основной Python-скрипт для запуска программы
├── video.mp4              # Пример видео для обработки
├── requirements.txt       # Список зависимостей Python
```

---

## Поддерживаемые классы

Модель обучена распознавать следующие 25 классов:

```plaintext
  - airplane
  - bear
  - bicycle
  - boat
  - bus
  - cat
  - cup
  - deer
  - elephant
  - horse
  - backpack
  - bird
  - bottle
  - car
  - cow
  - dog
  - handbag
  - motorcycle
  - person
  - sheep
  - suitcase
  - train
  - truck
  - umbrella
  - wine glass
```

---

## Требования

- **Ubuntu**: Проект тестировался на Ubuntu с настроенными драйверами NVIDIA.
- **CUDA 12.2**: Убедитесь, что у вас установлен драйвер NVIDIA версии **535.183.01** или новее, поддерживающий CUDA 12.2.
- **GPU**: NVIDIA GeForce RTX 4070 или аналогичная карта.
- **Python 3.10 или новее**

---

## Установка и настройка

### 1. Проверка CUDA

Перед началом работы убедитесь, что ваш драйвер и CUDA корректно установлены:
```bash
nvidia-smi
nvcc --version
```

Результат команды `nvidia-smi` должен содержать вашу видеокарту, драйвер и версию CUDA. Например:
```plaintext
Driver Version: 535.183.01   CUDA Version: 12.2
```

### 2. Установка зависимостей

1. Склонируйте репозиторий проекта:
   ```bash
   git clone https://github.com/your-repo/yolo-object-detection.git
   cd yolo-object-detection
   ```

2. Создайте виртуальное окружение и активируйте его:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

---

## Использование

Запустите основной файл программы и выберите режим работы:

1. Активируйте виртуальное окружение:
   ```bash
   source venv/bin/activate  # Для Linux/MacOS
   ```

2. Запустите `main.py`:
   ```bash
   python main.py
   ```

3. Выберите режим работы:
   - **1 - Веб-камера**: Для детекции объектов в реальном времени с веб-камеры.
   - **2 - Обработка видео**: Для обработки видеофайла.
   - **3 - Обработка изображений**: Для обработки изображений в указанной директории.

---

## Примеры использования

### Обработка видео
```bash
python main.py
# Выберите режим 2
Введите путь к видеофайлу: video.mp4
```

### Обработка изображений
```bash
python main.py
# Выберите режим 3
Введите путь к директории с изображениями: ./input_images
Введите путь для сохранения обработанных изображений: ./output_images
```

### Веб-камера
```bash
python main.py
# Выберите режим 1
```

---

## Примеры результатов

- **Обработка изображения**:
  Результаты сохраняются в папке `output_images/`.

- **Обработка видео**:
  Вы можете наблюдать результаты в реальном времени.

---

## Зависимости

Список зависимостей находится в файле `requirements.txt`. Пример содержимого:

```plaintext
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.5.5
numpy>=1.24.0
Pillow>=9.2.0
```

### Автор

- **Shakir Ramazanov**
- Контакты: [ramazanovshakir9@gmail.com]

---

Если потребуется дополнительная информация или улучшения, дайте знать!