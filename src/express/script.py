
import matplotlib.pyplot as plt
import gc
import concurrent.futures
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import torch.optim as optim
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from PIL import Image
import gc
import torch.nn as nn


num_classes = 24
learning_rate = 1e-4
class_name_to_idx = {'cartwheel': 0,
 'catch': 1,
 'clap': 2,
 'climb': 3,
 'dive': 4,
 'draw_sword': 5,
 'dribble': 6,
 'fencing': 7,
 'flic_flac': 8,
 'golf': 9,
 'handstand': 10,
 'hit': 11,
 'jump': 12,
 'pick': 13,
 'pour': 14,
 'pullup': 15,
 'push': 16,
 'pushup': 17,
 'shoot_ball': 18,
 'sit': 19,
 'situp': 20,
 'swing_baseball': 21,
 'sword_exercise': 22,
 'throw': 23}

# Создание модели
model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)


# Перенесите модель на GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = Compose([
    Resize([128, 171], antialias=True),
    CenterCrop([112, 112]),
    ToTensor(),
    Normalize(mean=[0.35117388, 0.4031328,  0.41368235], std=[0.20939734, 0.21085491, 0.21630637])
])


model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('action_recognition_model.pth'))
model.to(device)
model.eval()


def display_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Отключить оси координат для чистого отображения
    plt.show()

def process_frame(frame, model, transform, device, class_name_to_idx, frame_number, fps):
    # Обработка одного кадра в секунду
    if frame_number % int(fps) == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame)
            _, predicted = torch.max(outputs, 1)
            class_name = [name for name, idx in class_name_to_idx.items() if idx == predicted.item()][0]
            timestamp = frame_number / fps
            
            return class_name, timestamp
    else:
        return None


def process_frame_group(frame_group, model, transform, device, class_name_to_idx, timestamp, fps):
    processed_frames = []
    for frame in frame_group:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        processed_frames.append(frame)

    # Стекирование кадров и добавление батч-размерности
    frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(frames_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = [name for name, idx in class_name_to_idx.items() if idx == predicted.item()][0]
    
    del frames_tensor
    gc.collect()
    
    return class_name, timestamp


def process_video_parallel(video_path, model, transform, device, class_name_to_idx):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    action_timestamps = {class_name: [] for class_name in class_name_to_idx.keys()}
    
    max_workers = 4  # Ограничение количества одновременных задач
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Выборка трех кадров каждую секунду
    frames_to_process = [frames[i:i+3] for i in range(0, len(frames), int(fps)) if i+3 <= len(frames)]
    print('пошла жара')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {executor.submit(process_frame_group, frame_group, model, transform, device, class_name_to_idx, idx / fps, fps): idx for idx, frame_group in enumerate(frames_to_process)}
        for future in concurrent.futures.as_completed(future_to_frame):
            result = future.result()
            if result is not None:
                class_name, timestamp = result
                print(result)
                display_image(frames_to_process[future_to_frame[future]][0])
                action_timestamps[class_name].append(timestamp)

    return action_timestamps


def process_video(video_path, model, transform, device, class_name_to_idx):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    action_timestamps = {class_name: [] for class_name in class_name_to_idx.keys()}

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Выборка трех кадров каждую секунду
    frames_to_process = [frames[i:i+3] for i in range(0, len(frames), int(fps)) if i+3 <= len(frames)]
    del frames
    gc.collect()
    print('Обработка видео...')

    for idx, frame_group in enumerate(frames_to_process):
        result = process_frame_group(frame_group, model, transform, device, class_name_to_idx, idx / fps, fps)
        if result is not None:
            class_name, timestamp = result
            print(result)
            display_image(frame_group[0])  # Отображение первого кадра в группе
            action_timestamps[class_name].append(timestamp)

    return action_timestamps


video_path = 'vbn.mp4'
# Убедитесь, что модель находится в режиме .eval() и загружена на устройство перед вызовом этой функции
timestamps = process_video(video_path, model, transform, device, class_name_to_idx)

for action, times in timestamps.items():
    print(f"Действие '{action}' обнаружено в следующих временных метках: {times}")
