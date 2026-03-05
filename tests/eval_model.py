import argparse
import os
import random
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.models.model import EfficientNetB7


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def add_model_prefix_if_needed(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if any(k.startswith("model.") for k in state_dict.keys()):
        return state_dict
    return {f"model.{k}": v for k, v in state_dict.items()}


def load_model(device):
    model = EfficientNetB7().to(device)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = [
        os.path.join(project_root, "models", "efficientnet_b7_final.pth"),
        os.path.join(project_root, "models", "efficientnet.pth"),
        "models/efficientnet_b7_final.pth",
        "models/efficientnet.pth",
    ]

    errors = []
    loaded_from = None

    for model_path in model_paths:
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            continue

        try:
            checkpoint = torch.load(abs_path, map_location=device)
            state_dict = extract_state_dict(checkpoint)
            state_dict = strip_module_prefix(state_dict)
            candidates = [state_dict, add_model_prefix_if_needed(state_dict)]

            for candidate in candidates:
                try:
                    model.load_state_dict(candidate, strict=True)
                    loaded_from = abs_path
                    break
                except Exception:
                    pass

            if loaded_from:
                break
        except Exception as e:
            errors.append(f"{abs_path}: {e}")

    if not loaded_from:
        raise RuntimeError("Failed to load checkpoint.\n" + "\n".join(errors))

    model.eval()
    print(f"[OK] checkpoint loaded from: {loaded_from}")
    return model


class LabeledImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


def build_samples(train_dir: str):
    files = []
    for name in os.listdir(train_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            if name.startswith("dog."):
                files.append((os.path.join(train_dir, name), 1))
            elif name.startswith("cat."):
                files.append((os.path.join(train_dir, name), 0))
    return files


def stratified_split(samples, val_ratio=0.1, seed=42):
    dogs = [s for s in samples if s[1] == 1]
    cats = [s for s in samples if s[1] == 0]

    rng = random.Random(seed)
    rng.shuffle(dogs)
    rng.shuffle(cats)

    dog_val = int(len(dogs) * val_ratio)
    cat_val = int(len(cats) * val_ratio)

    val = dogs[:dog_val] + cats[:cat_val]
    train = dogs[dog_val:] + cats[cat_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def metrics(y_true, probs, threshold):
    y_pred = [1 if p > threshold else 0 for p in probs]
    tp, tn, fp, fn = confusion(y_true, y_pred)
    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate dogs-vs-cats checkpoint")
    parser.add_argument("--train-dir", default="data/dogsvscats/train")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    samples = build_samples(args.train_dir)
    if not samples:
        raise RuntimeError(f"No labeled files found in: {args.train_dir}")

    _, val_samples = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_val_samples and args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    print(f"val_samples={len(val_samples)}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = LabeledImageDataset(val_samples, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_model(device)

    y_true = []
    probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).squeeze(1)
            probs.extend(out.detach().cpu().tolist())
            y_true.extend(y.int().tolist())

    m05 = metrics(y_true, probs, 0.5)

    best_t = 0.5
    best_m = m05
    for t_i in range(1, 100):
        t = t_i / 100.0
        m = metrics(y_true, probs, t)
        if m["acc"] > best_m["acc"]:
            best_t = t
            best_m = m

    print("\n=== threshold=0.50 ===")
    print(f"acc={m05['acc']:.4f} precision={m05['precision']:.4f} recall={m05['recall']:.4f} f1={m05['f1']:.4f}")
    print(f"confusion(tp,tn,fp,fn)=({m05['tp']},{m05['tn']},{m05['fp']},{m05['fn']})")

    print("\n=== best threshold (by acc) ===")
    print(f"threshold={best_t:.2f}")
    print(f"acc={best_m['acc']:.4f} precision={best_m['precision']:.4f} recall={best_m['recall']:.4f} f1={best_m['f1']:.4f}")
    print(f"confusion(tp,tn,fp,fn)=({best_m['tp']},{best_m['tn']},{best_m['fp']},{best_m['fn']})")


if __name__ == "__main__":
    main()
