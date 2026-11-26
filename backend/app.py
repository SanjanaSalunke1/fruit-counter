import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
detector = YOLO(str(MODEL_DIR / "fruitcountY11m.pt"))
CROP_MARGIN = float(os.getenv("CROP_MARGIN", "0.15"))
if not 0 <= CROP_MARGIN < 0.5:
    raise ValueError("CROP_MARGIN must be at least 0 and less than 0.5")

# The checkpoint is a torchvision EfficientNetV2-S state_dict. The JSON maps
# each output index to its label; COLOR_CLASSES can override it for deployment.
COLOR_CLASSES_PATH = MODEL_DIR / "bell_pepper_color_classes.json"
try:
    class_mapping = json.loads(COLOR_CLASSES_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_mapping, dict) or set(class_mapping) != {
        str(index) for index in range(len(class_mapping))
    }:
        raise ValueError("class mapping keys must be contiguous indices starting at 0")
    configured_classes = [class_mapping[str(index)] for index in range(len(class_mapping))]
except (OSError, json.JSONDecodeError, ValueError) as error:
    raise RuntimeError(f"Invalid color class mapping: {COLOR_CLASSES_PATH}") from error

COLOR_CLASSES = [
    label.strip().lower()
    for label in os.getenv("COLOR_CLASSES", ",".join(configured_classes)).split(",")
    if label.strip()
]
if len(COLOR_CLASSES) != 3:
    raise RuntimeError("COLOR_CLASSES must contain exactly three comma-separated labels")

COLOR_MODEL_PATH = MODEL_DIR / "best_bell_pepper_color_effnetv2s.pth"
if not COLOR_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Bell pepper color model not found: {COLOR_MODEL_PATH}. "
        "Place best_bell_pepper_color_effnetv2s.pth in backend/model."
    )

color_model = models.efficientnet_v2_s(weights=None)
color_model.classifier[1] = torch.nn.Linear(color_model.classifier[1].in_features, len(COLOR_CLASSES))
color_state = torch.load(COLOR_MODEL_PATH, map_location="cpu")
# Accept raw state_dicts and common training checkpoint wrappers.
if isinstance(color_state, dict):
    for state_key in ("state_dict", "model_state_dict", "model"):
        if state_key in color_state and isinstance(color_state[state_key], dict):
            color_state = color_state[state_key]
            break
if not isinstance(color_state, dict):
    raise RuntimeError(f"Unsupported color model checkpoint format: {COLOR_MODEL_PATH}")
color_state = {
    key.removeprefix("module."): value
    for key, value in color_state.items()
}
try:
    color_model.load_state_dict(color_state)
except RuntimeError as error:
    raise RuntimeError(
        f"Could not load {COLOR_MODEL_PATH.name}. Confirm this is a torchvision "
        "EfficientNetV2-S checkpoint and COLOR_CLASSES matches its output classes."
    ) from error
color_model.eval()

# Maturity classifier: YOLO11n-cls trained on VegNet bell peppers (see maturity/).
# Ripe = fresh pepper, Overripe = old (wrinkled) or dried pepper.
MATURITY_MODEL_PATH = MODEL_DIR / "maturity_yolo11n_cls.pt"
if not MATURITY_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Maturity model not found: {MATURITY_MODEL_PATH}. "
        "Train it with maturity/train.py or place maturity_yolo11n_cls.pt in backend/model."
    )
maturity_model = YOLO(str(MATURITY_MODEL_PATH))
MATURITY_CLASSES = [maturity_model.names[index] for index in sorted(maturity_model.names)]
# Unlike the colour crop (trimmed inward), maturity needs the whole fruit outline
# (wrinkles, shrivelled shape), so its crop is padded outward instead.
MATURITY_PAD = float(os.getenv("MATURITY_PAD", "0.08"))

color_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _dominant_center_hsv_color(crop_bgr):
    """Return a color vote only when the crop center has a clear HSV majority."""
    height, width = crop_bgr.shape[:2]
    inset_x, inset_y = int(width * 0.15), int(height * 0.15)
    center = crop_bgr[inset_y:height - inset_y, inset_x:width - inset_x]
    if center.size == 0:
        return None

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation >= 60) & (value >= 35)
    red = ((hue <= 10) | (hue >= 170)) & colorful
    # Green peppers can shift toward yellow-green under bright greenhouse light.
    yellow = (hue >= 11) & (hue <= 30) & colorful
    green = (hue >= 31) & (hue <= 95) & colorful
    votes = {
        "red": int(np.count_nonzero(red)),
        "yellow": int(np.count_nonzero(yellow)),
        "green": int(np.count_nonzero(green)),
    }
    total = sum(votes.values())
    if total < 30:
        return None

    ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    color, count = ranked[0]
    second_count = ranked[1][1]
    share = count / total
    if share < 0.55 or count < second_count * 1.35:
        return None
    return color, share


def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = color_transform(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(color_model(tensor)[0], dim=0)
    class_index = int(torch.argmax(probabilities).item())
    model_color = COLOR_CLASSES[class_index]
    model_confidence = float(probabilities[class_index].item())

    # Correct obvious model/crop disagreements using the crop's dominant
    # saturated color; keep the EfficientNet prediction when the vote is mixed.
    hsv_vote = _dominant_center_hsv_color(crop_bgr)
    if hsv_vote is not None and hsv_vote[0] != model_color:
        return hsv_vote
    return model_color, model_confidence


def classify_maturity(image_bgr, x1, y1, x2, y2):
    """Ripe / Overripe for one detected pepper. Ultralytics reads numpy arrays as BGR."""
    height, width = image_bgr.shape[:2]
    pad_x, pad_y = int((x2 - x1) * MATURITY_PAD), int((y2 - y1) * MATURITY_PAD)
    crop = image_bgr[max(0, y1 - pad_y):min(height, y2 + pad_y),
                     max(0, x1 - pad_x):min(width, x2 + pad_x)]
    result = maturity_model(crop, imgsz=224, verbose=False)[0]
    class_index = int(result.probs.top1)
    return result.names[class_index], float(result.probs.top1conf.item())


@app.post("/count")
async def count_fruits(file: UploadFile = File(...), min_conf: float = 0.35):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    height, width = image.shape[:2]
    results = detector(image, conf=min_conf, iou=0.45, verbose=False)
    detections = results[0].boxes
    boxes = []
    color_counts = {label: 0 for label in COLOR_CLASSES}
    maturity_counts = {label: 0 for label in MATURITY_CLASSES}

    if detections is not None and len(detections):
        coords_list = detections.xyxy.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        for coords, detection_confidence in zip(coords_list, confidences):
            x1, y1, x2, y2 = [int(round(value)) for value in coords]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Trim the YOLO box inward on every side before color classification.
            crop_width, crop_height = x2 - x1, y2 - y1
            margin_x = int(crop_width * CROP_MARGIN)
            margin_y = int(crop_height * CROP_MARGIN)
            inner_x1 = max(0, min(width, x1 + margin_x))
            inner_x2 = max(0, min(width, x2 - margin_x))
            inner_y1 = max(0, min(height, y1 + margin_y))
            inner_y2 = max(0, min(height, y2 - margin_y))

            # Very small boxes may collapse after trimming; use their valid
            # clamped YOLO crop in that case so classification still works.
            if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
                inner_x1, inner_y1, inner_x2, inner_y2 = x1, y1, x2, y2

            color, color_confidence = classify_crop(
                image[inner_y1:inner_y2, inner_x1:inner_x2]
            )
            color_counts[color] += 1
            maturity, maturity_confidence = classify_maturity(image, x1, y1, x2, y2)
            maturity_counts[maturity] += 1
            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(detection_confidence),
                "color": color,
                "color_confidence": color_confidence,
                "maturity": maturity,
                "maturity_confidence": maturity_confidence,
            })

    return {
        "fruit_count": len(boxes),
        "total_bell_pepper_count": len(boxes),
        "color_counts": color_counts,
        "maturity_counts": maturity_counts,
        "boxes": boxes,
    }


@app.get("/")
async def health_check():
    return {"status": "ok"}
