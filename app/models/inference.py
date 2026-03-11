import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

MODEL_REPO_ID = "qkek033/Dogs_Cats_Image_Classification"
MODEL_NAME = "model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__version__ = "5.0.0"

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

def download_model_from_hub() -> str:
    """HuggingFace Hub에서 모델 다운로드"""
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_NAME,
            cache_dir=str(Path.home() / '.cache' / 'huggingface')
        )
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace Hub: {e}")

def load_model(model_path: str):
    """SimpleCNN 모델 로드"""
    try:
        model = SimpleCNN(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def get_model():
    """모델 로드 (싱글톤 패턴)"""
    if not hasattr(get_model, '_model'):
        model_path = download_model_from_hub()
        get_model._model = load_model(model_path)
    return get_model._model

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """이미지 전처리"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(img).unsqueeze(0).to(device)

def resize_array(array, size=(128, 128)):
    """NumPy 배열 크기 조정 (cv2 없이)"""
    img = Image.fromarray((array * 255).astype('uint8'), mode='L')
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.array(img) / 255.0

def generate_grad_cam(model, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM 생성"""
    input_tensor.requires_grad = True
    
    output = model(input_tensor)
    
    if class_idx >= output.size(1):
        class_idx = output.argmax(dim=1).item()
    
    class_output = output[0, class_idx]
    
    model.zero_grad()
    class_output.backward()
    
    gradients = input_tensor.grad.data[0].cpu().numpy()
    
    if gradients.max() - gradients.min() > 1e-5:
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    
    cam = np.mean(gradients, axis=0)
    cam = np.maximum(cam, 0)
    
    if cam.max() > 0:
        cam = cam / cam.max()
    
    cam = resize_array(cam, (128, 128))
    
    return cam

def predict_image(image_bytes: bytes):
    """이미지 예측"""
    try:
        input_tensor = preprocess_image(image_bytes)
        
        model = get_model()
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        confidence_score = confidence.item()
        
        labels = ["cat", "dog"]
        label = labels[class_idx]
        
        # 두 클래스의 확률 확인
        cat_prob = probabilities[0, 0].item()
        dog_prob = probabilities[0, 1].item()
        
        # 조건:
        # 1. 고양이 또는 강아지 중 하나가 70% 이상이어야 함
        # 2. 두 클래스의 차이가 30% 이상이어야 함
        max_prob = max(cat_prob, dog_prob)
        margin = max_prob - min(cat_prob, dog_prob)
        
        rejected = (max_prob < 0.70) or (margin < 0.30)
        reject_reason = "not_animal" if rejected else None
        
        # 거부된 경우 Grad-CAM 생성 안 함
        cam = None
        if not rejected:
            cam = generate_grad_cam(model, input_tensor, class_idx)
        
        return label, confidence_score, cam, rejected, reject_reason
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
