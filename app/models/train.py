import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from app.data.preprocess import load_data, split_data
from app.data.dataset import DogsCatsDataset, get_transforms
from app.models.model import SimpleCNN


# ------------------------
# 설정
# ------------------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------
# 데이터 준비
# ------------------------
print(" 데이터 로딩 중...")
train_files, test_files, labels = load_data()
train_list, val_list = split_data(train_files, labels)

train_transform, val_transform = get_transforms()

train_dataset = DogsCatsDataset(train_list, transform=train_transform)
val_dataset = DogsCatsDataset(val_list, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} | Valid: {len(val_dataset)}")


# ------------------------
# 모델
# ------------------------
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------------------
# 학습 함수
# ------------------------
def train_one_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ------------------------
# 검증 함수
# ------------------------
def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ------------------------
# 학습 루프
# ------------------------
print(" 학습 시작")

for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_acc = validate()

    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


# ------------------------
# 모델 저장
# ------------------------
import os
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/model.pth")
print(" 모델 저장 완료 (models/model.pth)")