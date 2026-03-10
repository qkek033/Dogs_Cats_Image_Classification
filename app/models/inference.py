import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

MODEL_REPO_ID = "qkek033/dogs-cats-classifier"
EFFICIENTNET_MODEL_NAME = "efficientnet_b7_final.pth"
VIT_MODEL_NAME = "vit_base_patch32_final.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_model_from_hub(model_name: str) -> str:
    """HuggingFace Hub에서 모델 다운로드"""
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=model_name,
            cache_dir=str(Path.home() / '.cache' / 'huggingface')
        )
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name} from HuggingFace Hub: {e}")

def load_efficientnet_model(model_path: str):
    """EfficientNet 모델 로드"""
    try:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load EfficientNet model: {e}")

def load_vit_model(model_path: str):
    """Vision Transformer 모델 로드"""
    try:
        from timm import create_model
        model = create_model('vit_base_patch32_224', num_classes=2, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ViT model: {e}")

def get_efficientnet_model():
    """EfficientNet 모델 (싱글톤 패턴)"""
    if not hasattr(get_efficientnet_model, '_model'):
        model_path = download_model_from_hub(EFFICIENTNET_MODEL_NAME)
        get_efficientnet_model._model = load_efficientnet_model(model_path)
    return get_efficientnet_model._model

def get_vit_model():
    """ViT 모델 (싱글톤 패턴)"""
    if not hasattr(get_vit_model, '_model'):
        model_path = download_model_from_hub(VIT_MODEL_NAME)
        get_vit_model._model = load_vit_model(model_path)
    return get_vit_model._model

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """이미지 전처리"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(img).unsqueeze(0).to(device)

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
    
    cam = cv2.resize(cam, (224, 224))
    
    return cam

def predict_image(image_bytes: bytes):
    """이미지 예측"""
    try:
        input_tensor = preprocess_image(image_bytes)
        
        model = get_efficientnet_model()
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        confidence_score = confidence.item()
        
        labels = ["cat", "dog"]
        label = labels[class_idx]
        
        cam = generate_grad_cam(model, input_tensor, class_idx)
        
        rejected = confidence_score < 0.5
        reject_reason = "low_confidence" if rejected else None
        
        return label, confidence_score, cam, rejected, reject_reason
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
