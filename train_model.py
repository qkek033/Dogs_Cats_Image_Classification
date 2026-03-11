"""
SimpleCNN 모델 학습 스크립트
Kaggle Dogs vs Cats Redux 데이터셋 사용
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from pathlib import Path

# 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = "data/train"  # 데이터셋 디렉토리
MODEL_SAVE_PATH = "models/model.pth"

print(f"사용 디바이스: {DEVICE}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    """모델 학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Valid - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
    
    return model

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 디렉토리가 없습니다: {DATA_DIR}")
        print("\n다음 방법으로 데이터를 준비해주세요:")
        print("1. Kaggle에서 'Dogs vs Cats Redux' 데이터 다운로드")
        print("   https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition")
        print("\n2. 데이터 구조:")
        print("   data/train/")
        print("   ├── cats/")
        print("   │   ├── cat.1.jpg")
        print("   │   ├── cat.2.jpg")
        print("   │   └── ...")
        print("   └── dogs/")
        print("       ├── dog.1.jpg")
        print("       ├── dog.2.jpg")
        print("       └── ...")
        return
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 데이터셋 로드
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ 데이터 로드 완료")
    print(f"   전체: {len(full_dataset)}, 학습: {len(train_dataset)}, 검증: {len(val_dataset)}")
    
    # 모델 생성
    model = SimpleCNN(num_classes=2).to(DEVICE)
    
    # 학습
    print("\n🚀 모델 학습 시작...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    
    # 모델 저장
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ 모델 저장 완료: {MODEL_SAVE_PATH}")
    
    print("\n📤 다음 단계: HuggingFace Hub에 모델 업로드")
    print("   python upload_to_hub.py")

if __name__ == "__main__":
    main()
