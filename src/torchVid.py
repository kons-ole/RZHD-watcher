import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_video_dataset import VideoDataset  # Подразумевается, что вы используете созданный в предыдущем ответе пользовательский датасет.

# Определение сверточной нейронной сети
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 10 * 10, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Параметры обучения
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_classes = 2  # Допустим, у вас есть два класса: "езда на велосипеде" и "не езда на велосипеде".

# Создание датасета и загрузчика данных
transform = transforms.Compose([transforms.ToPILImage(),  # Преобразование кадра в изображение
                                transforms.Resize((224, 224)),  # Изменение размера кадра
                                transforms.ToTensor()])  # Преобразование в тензор PyTorch

dataset = VideoDataset('../data/Biking', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Создание модели и оптимизатора
model = CNNModel(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Обучение модели
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data  # Входные видеокадры
        labels = torch.tensor([0 if "Biking" in path else 1 for path in data])  # 0 - "езда на велосипеде", 1 - "не езда на велосипеде"

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Сохранение модели
torch.save(model.state_dict(), 'biking_model.pth')
