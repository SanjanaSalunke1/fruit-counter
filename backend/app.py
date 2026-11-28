import heapq
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

# Keep inference self-contained: no Ultralytics usage events or runtime pip
# installs. Must be set before Ultralytics is imported.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from ultralytics import YOLO  # noqa: E402

try:
    import openvino as ov
except ImportError:  # PyTorch inference still works without OpenVINO.
    ov = None

logger = logging.getLogger("fruit_counter")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False
if not logger.handlers:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    file_handler = RotatingFileHandler(
        log_dir / "fruit_counter.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

app = FastAPI(title="Fruit Counter API", version="1.1.0")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check backend logs with the request ID.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    # Errors stand out in the terminal: 4xx as WARNING, 5xx as ERROR.
    level = logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.1f", request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log why a request was rejected (missing file, out-of-range min_conf, ...)."""
    reasons = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )
    logger.warning("request_invalid request_id=%s path=%s reason=%s", request.state.request_id, request.url.path, reasons)
    return await request_validation_exception_handler(request, exc)


app.add_middleware(
    CORSMiddleware,
    # Public, cookie-free API: any origin may call it, so credentials stay off.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(250 * 1024 * 1024)))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "300"))
# Frames are sampled by time so high-frame-rate clips do not multiply detector
# work. VIDEO_FRAME_STRIDE, when set, forces a fixed stride instead.
VIDEO_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "10"))
if VIDEO_SAMPLE_FPS <= 0:
    raise ValueError("VIDEO_SAMPLE_FPS must be greater than 0")
VIDEO_FRAME_STRIDE = max(1, int(os.environ["VIDEO_FRAME_STRIDE"])) if os.getenv("VIDEO_FRAME_STRIDE") else None
# Each video track is classified from at most this many of its best sightings.
VIDEO_TRACK_SAMPLES = 5
CLASSIFY_BATCH_SIZE = 32
CROP_MARGIN = float(os.getenv("CROP_MARGIN", "0.15"))
if not 0 <= CROP_MARGIN < 0.5:
    raise ValueError("CROP_MARGIN must be at least 0 and less than 0.5")

MODEL_DIR = Path(__file__).resolve().parent / "model"
DETECTOR_PATH = MODEL_DIR / "fruitcountY11m.pt"
# Generated from DETECTOR_PATH on startup; re-exported whenever the .pt is newer.
DETECTOR_OPENVINO_PATH = MODEL_DIR / "fruitcountY11m_openvino_model"


def _load_detector():
    """Return (detector, device), preferring an FP32 CPU OpenVINO export.

    The export keeps dynamic input shapes so Ultralytics applies the same
    rectangular letterbox as PyTorch, which yields identical detections.
    Any OpenVINO problem falls back to the PyTorch model.
    """
    warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    if ov is not None:
        try:
            if (not DETECTOR_OPENVINO_PATH.is_dir()
                    or DETECTOR_OPENVINO_PATH.stat().st_mtime < DETECTOR_PATH.stat().st_mtime):
                logger.info("detector_openvino_export_started path=%s", DETECTOR_OPENVINO_PATH)
                YOLO(str(DETECTOR_PATH)).export(format="openvino", half=False, int8=False, dynamic=True, imgsz=640)
            model = YOLO(str(DETECTOR_OPENVINO_PATH), task="detect")
            # "intel:cpu" pins the CPU; Ultralytics otherwise uses AUTO, which
            # can hand inference to a GPU at reduced precision.
            model(warmup_frame, device="intel:cpu", verbose=False)
            compiled = model.predictor.model.backend.ov_compiled_model
            devices = list(compiled.get_property("EXECUTION_DEVICES"))
            precision = compiled.get_property("INFERENCE_PRECISION_HINT")
            if devices == ["CPU"] and precision == ov.Type.f32:
                logger.info("detector_loaded backend=openvino devices=%s precision=%s", devices, precision)
                return model, "intel:cpu"
            logger.warning("detector_openvino_rejected devices=%s precision=%s", devices, precision)
        except Exception:
            logger.exception("detector_openvino_failed; using PyTorch")
    model = YOLO(str(DETECTOR_PATH))
    model(warmup_frame, verbose=False)
    logger.info("detector_loaded backend=pytorch")
    return model, None


detector, detector_device = _load_detector()
inference_lock = threading.Lock()


def detect(frame, min_conf):
    """Run the detector with the settings shared by the photo and video endpoints."""
    return detector(frame, conf=min_conf, iou=0.45, device=detector_device, verbose=False)[0]


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

color_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _compile_classifier(model, image_size, name):
    """Return a batch -> logits callable: OpenVINO FP32 on CPU, else the PyTorch model.

    OpenVINO matches PyTorch to ~1e-6 in probability; f32 is forced so CPUs with
    bf16 support do not lower precision.
    """
    if ov is not None:
        try:
            converted = ov.convert_model(
                model,
                example_input=torch.zeros(1, 3, image_size, image_size),
                input=[(-1, 3, image_size, image_size)],
            )
            compiled = ov.Core().compile_model(
                converted, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
            )
            logger.info("classifier_loaded name=%s backend=openvino", name)
            # Copy: OpenVINO reuses its output buffer on the next inference.
            return lambda batch: torch.from_numpy(compiled(batch.numpy())[0].copy())
        except Exception:
            logger.exception("classifier_openvino_failed name=%s; using PyTorch", name)
    logger.info("classifier_loaded name=%s backend=pytorch", name)
    return model


color_infer = _compile_classifier(color_model, 224, "color")
logger.info(
    "models_ready detector=%s color_model=%s color_classes=%s video_sampling=%s max_video_mb=%d max_video_seconds=%d",
    "openvino" if detector_device else "pytorch",
    "pytorch" if color_infer is color_model else "openvino",
    COLOR_CLASSES,
    f"stride={VIDEO_FRAME_STRIDE}" if VIDEO_FRAME_STRIDE else f"{VIDEO_SAMPLE_FPS:g}fps",
    MAX_VIDEO_BYTES // (1024 * 1024), MAX_VIDEO_SECONDS,
)


def _dominant_center_hsv_color(crop_bgr):
    """Return a color vote only when the crop center has a clear HSV majority."""
    height, width = crop_bgr.shape[:2]
    inset_x, inset_y = int(width * 0.15), int(height * 0.15)
    center = crop_bgr[inset_y:height - inset_y, inset_x:width - inset_x]
    if center.size == 0:
        return None

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation >= 45) & (value >= 30)
    # Dark olive/brown peppers often land in the yellow hue band. Keep yellow
    # for clearly bright yellow skin; map darker brown-red tones to red and
    # dark olive tones to green instead.
    red = ((hue <= 10) | (hue >= 170) | ((hue > 10) & (hue < 23) & (value < 105))) & colorful
    # True yellow skin sits at OpenCV hue ~20-28 and is vivid. Sunlit or light
    # green skin shifts to hue ~29-40, and olive green reaches the yellow band
    # only with weak saturation, so both stay out of the yellow vote.
    yellow = (hue >= 20) & (hue <= 28) & (
        ((saturation >= 140) & (value >= 105)) | ((saturation >= 90) & (value >= 170))
    )
    green = (hue >= 22) & (hue <= 95) & colorful & ~yellow & ~red
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


def _batched_probabilities(model, transform, crops_bgr):
    """Run a classifier over crops in fixed-size batches; return (n, classes) probabilities."""
    outputs = []
    # Compiled OpenVINO models share one infer request, so calls are serialized.
    with inference_lock, torch.inference_mode():
        for start in range(0, len(crops_bgr), CLASSIFY_BATCH_SIZE):
            batch = torch.stack([
                transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                for crop in crops_bgr[start:start + CLASSIFY_BATCH_SIZE]
            ])
            outputs.append(torch.softmax(model(batch), dim=1))
    return torch.cat(outputs) if outputs else torch.empty((0, 0))


def _resolve_color(crop_bgr, probabilities):
    class_index = int(torch.argmax(probabilities).item())
    model_color = COLOR_CLASSES[class_index]
    model_confidence = float(probabilities[class_index].item())

    # Correct obvious model/crop disagreements using the crop's dominant
    # saturated color; keep the EfficientNet prediction when the vote is mixed.
    hsv_vote = _dominant_center_hsv_color(crop_bgr)
    if hsv_vote is not None:
        if hsv_vote[0] != model_color:
            return hsv_vote
        return model_color, model_confidence
    # Do not trust the color model's yellow label without a clear bright-yellow
    # pixel vote. For ambiguous dark/olive crops, choose its stronger green/red
    # alternative instead of returning yellow.
    if model_color == "yellow":
        alternatives = [
            index for index, label in enumerate(COLOR_CLASSES)
            if label in {"green", "red"}
        ]
        if alternatives:
            alternative_index = max(alternatives, key=lambda index: probabilities[index].item())
            return COLOR_CLASSES[alternative_index], float(probabilities[alternative_index].item())
    return model_color, model_confidence


def classify_crops(crops_bgr):
    """Return (color, confidence) for each color crop."""
    probabilities = _batched_probabilities(color_infer, color_transform, crops_bgr)
    return [_resolve_color(crop, row) for crop, row in zip(crops_bgr, probabilities)]


def _crop_box(frame, box):
    """Clamp a YOLO box; return (x1, y1, x2, y2, color_crop) or None.

    The color crop is trimmed inward to avoid background leaves.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # Trim the YOLO box inward on every side before color classification.
    margin_x = int((x2 - x1) * CROP_MARGIN)
    margin_y = int((y2 - y1) * CROP_MARGIN)
    inner_x1, inner_x2 = x1 + margin_x, x2 - margin_x
    inner_y1, inner_y2 = y1 + margin_y, y2 - margin_y
    # Very small boxes may collapse after trimming; use their valid
    # clamped YOLO crop in that case so classification still works.
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        inner_x1, inner_y1, inner_x2, inner_y2 = x1, y1, x2, y2
    return x1, y1, x2, y2, frame[inner_y1:inner_y2, inner_x1:inner_x2]


@app.post("/count")
async def count_fruits(file: UploadFile = File(...), min_conf: float = Query(0.35, ge=0, le=1)):
    logger.info("image_count_started filename=%s min_conf=%.2f", file.filename, min_conf)
    started = time.perf_counter()
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("image_rejected filename=%s reason=invalid_image bytes=%d", file.filename, len(image_bytes))
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    with inference_lock:
        detections = detect(image, min_conf).boxes
    boxes = []
    color_counts = {label: 0 for label in COLOR_CLASSES}

    if detections is not None and len(detections):
        coords_list = detections.xyxy.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        kept = [
            (cropped, float(detection_confidence))
            for coords, detection_confidence in zip(coords_list, confidences)
            if (cropped := _crop_box(image, coords)) is not None
        ]
        # One batched classifier pass instead of one call per pepper.
        colors = classify_crops([cropped[4] for cropped, _ in kept])
        for ((x1, y1, x2, y2, _), detection_confidence), (color, color_confidence) in zip(kept, colors):
            color_counts[color] += 1
            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(detection_confidence),
                "color": color,
                "color_confidence": color_confidence,
            })

    response = {
        "fruit_count": len(boxes),
        "total_bell_pepper_count": len(boxes),
        "color_counts": color_counts,
        "boxes": boxes,
    }
    logger.info(
        "image_count_complete filename=%s fruit_count=%d color_counts=%s duration_ms=%.0f",
        file.filename, len(boxes), color_counts, (time.perf_counter() - started) * 1000,
    )
    return response


def _estimate_camera_transform(previous_gray, current_gray):
    """Estimate current-frame to previous-frame motion from stationary scene texture."""
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if previous_points is None or len(previous_points) < 12:
        return None
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if current_points is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    if np.count_nonzero(valid) < 10:
        return None
    current = current_points.reshape(-1, 2)[valid]
    previous = previous_points.reshape(-1, 2)[valid]
    transform, inliers = cv2.findHomography(current, previous, cv2.RANSAC, 3.0)
    if transform is None or inliers is None:
        return None
    inlier_ratio = float(np.count_nonzero(inliers)) / len(inliers)
    transform = transform / transform[2, 2]
    if inlier_ratio < 0.25 or not np.isfinite(transform).all():
        return None
    # Reject implausible per-sample jumps that would corrupt the cumulative map.
    if abs(transform[0, 2]) > current_gray.shape[1] * 0.8 or abs(transform[1, 2]) > current_gray.shape[0] * 0.8:
        return None
    return transform


def _map_box_to_scene(box, transform):
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(corners, transform)[0]
    return np.array([
        mapped[:, 0].min(), mapped[:, 1].min(),
        mapped[:, 0].max(), mapped[:, 1].max(),
    ], dtype=np.float32)


def _intersection_over_union(box_a, box_b):
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _merge_scene_detections(boxes, confidences, scene_tracks):
    """Merge sightings at the same stabilized scene location, one detection per track/frame."""
    candidates = []
    for track_id, track in enumerate(scene_tracks):
        old = track["box"]
        old_area = max(1.0, (old[2] - old[0]) * (old[3] - old[1]))
        old_center = np.array([(old[0] + old[2]) / 2, (old[1] + old[3]) / 2])
        old_diagonal = max(1.0, float(np.hypot(old[2] - old[0], old[3] - old[1])))
        for box_index, box in enumerate(boxes):
            new_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            area_ratio = max(old_area, new_area) / min(old_area, new_area)
            if area_ratio > 3.0:
                continue
            overlap = _intersection_over_union(old, box)
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            distance = float(np.linalg.norm(old_center - center))
            diagonal = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
            proximity = max(0.0, 1.0 - distance / (0.55 * max(old_diagonal, diagonal)))
            if overlap >= 0.12 or (area_ratio < 2.0 and proximity >= 0.55):
                candidates.append((0.7 * overlap + 0.3 * proximity, track_id, box_index))

    used_tracks, used_boxes = set(), set()
    assignments = {}
    for _, track_id, box_index in sorted(candidates, reverse=True):
        if track_id in used_tracks or box_index in used_boxes:
            continue
        track = scene_tracks[track_id]
        track["box"] = 0.7 * track["box"] + 0.3 * boxes[box_index]
        track["observations"] += 1
        track["confidence"] = max(track["confidence"], float(confidences[box_index]))
        used_tracks.add(track_id)
        used_boxes.add(box_index)
        assignments[box_index] = track_id

    for box_index, box in enumerate(boxes):
        if box_index not in assignments:
            track_id = len(scene_tracks)
            scene_tracks.append({
                "box": box.copy(), "observations": 1,
                "confidence": float(confidences[box_index]), "samples": [],
            })
            assignments[box_index] = track_id
    return assignments


@app.post("/count-video")
async def count_video(request: Request, file: UploadFile = File(...), min_conf: float = Query(0.35, ge=0, le=1)):
    """Count peppers across a moving-camera clip with scene-motion compensation."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        logger.warning("video_rejected filename=%s reason=unsupported_format", file.filename)
        raise HTTPException(status_code=400, detail="Unsupported video format")

    logger.info("video_count_started filename=%s min_conf=%.2f", file.filename, min_conf)
    started = time.perf_counter()

    temp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    logger.warning("video_rejected filename=%s reason=size_limit bytes=%d", file.filename, size)
                    raise HTTPException(status_code=413, detail="Video exceeds the upload size limit")
                temp_file.write(chunk)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.warning("video_rejected filename=%s reason=unreadable", file.filename)
            raise HTTPException(status_code=400, detail="Could not read the uploaded video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        if duration > MAX_VIDEO_SECONDS:
            logger.warning("video_rejected filename=%s reason=duration_limit seconds=%.2f", file.filename, duration)
            raise HTTPException(status_code=413, detail=f"Video must be {MAX_VIDEO_SECONDS} seconds or shorter")
        frame_stride = VIDEO_FRAME_STRIDE or (max(1, round(fps / VIDEO_SAMPLE_FPS)) if fps > 0 else 3)

        scene_tracks = []
        sample_order = 0
        frame_detection_counts = []
        previous_gray = None
        global_transform = np.eye(3, dtype=np.float64)
        registration_successes = 0
        registration_failures = 0
        frame_index = 0
        processed_frames = 0
        # Serialize access to the shared detector while processing this clip.
        with inference_lock:
            while True:
                # Skipped frames are only grabbed, not converted to images.
                if frame_index % frame_stride:
                    if not cap.grab():
                        break
                    frame_index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                width = frame.shape[1]
                registration_scale = min(1.0, 640 / width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if registration_scale < 1.0:
                    gray = cv2.resize(gray, None, fx=registration_scale, fy=registration_scale, interpolation=cv2.INTER_AREA)
                if previous_gray is not None:
                    relative_transform = _estimate_camera_transform(previous_gray, gray)
                    if relative_transform is not None:
                        global_transform = global_transform @ relative_transform
                        global_transform /= global_transform[2, 2]
                        registration_successes += 1
                    else:
                        registration_failures += 1
                previous_gray = gray

                result = detect(frame, min_conf)
                processed_frames += 1
                if result.boxes is not None and len(result.boxes):
                    coords = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().tolist()
                else:
                    coords = np.empty((0, 4), dtype=np.float32)
                    confidences = []
                frame_detection_counts.append(len(coords))
                scene_boxes = [
                    _map_box_to_scene(box * registration_scale, global_transform)
                    for box in coords
                ]
                assignments = _merge_scene_detections(scene_boxes, confidences, scene_tracks)
                for detection_index, box in enumerate(coords):
                    track = scene_tracks[assignments[detection_index]]
                    # Keep the most confident, spaced-out sightings; classify after tracking
                    # so only confirmed tracks reach the color classifier, in batches.
                    if track["observations"] == 1 or track["observations"] % 3 == 0:
                        cropped = _crop_box(frame, box)
                        if cropped is not None:
                            sample_order += 1
                            # Copy so stored crops do not keep whole frames alive.
                            sample = (confidences[detection_index], sample_order, cropped[4].copy())
                            if len(track["samples"]) < VIDEO_TRACK_SAMPLES:
                                heapq.heappush(track["samples"], sample)
                            elif sample[0] > track["samples"][0][0]:
                                heapq.heapreplace(track["samples"], sample)
                frame_index += 1

        color_counts = {label: 0 for label in COLOR_CLASSES}
        confirmed_tracks = [
            track for track in scene_tracks
            if track["observations"] >= 2 or track["confidence"] >= min(0.95, min_conf + 0.25)
        ]
        count = len(confirmed_tracks)
        median_detections = 0
        maximum_detections = 0
        if frame_detection_counts:
            sorted_counts = sorted(frame_detection_counts)
            median_detections = int(np.median(sorted_counts))
            maximum_detections = sorted_counts[-1]
        samples = [
            (track_index, sample)
            for track_index, track in enumerate(confirmed_tracks)
            for sample in track["samples"]
        ]
        sample_colors = classify_crops([sample[2] for _, sample in samples])
        color_votes = [{} for _ in confirmed_tracks]
        for (track_index, _), (color, _) in zip(samples, sample_colors):
            color_votes[track_index][color] = color_votes[track_index].get(color, 0) + 1
        for votes in color_votes:
            # Tracks never sampled (all crops degenerate) default to the first class.
            color = max(votes, key=votes.get) if votes else COLOR_CLASSES[0]
            color_counts[color] += 1
        logger.info(
            "video_count_complete filename=%s fruit_count=%d color_counts=%s duration_ms=%.0f scene_tracks=%d frames=%d median_frame_count=%d max_frame_count=%d camera_registration_successes=%d camera_registration_failures=%d",
            file.filename, count, color_counts, (time.perf_counter() - started) * 1000,
            len(scene_tracks), processed_frames, median_detections,
            maximum_detections, registration_successes, registration_failures,
        )
        return {
            "fruit_count": count,
            "total_bell_pepper_count": count,
            "color_counts": color_counts,
            "counting_method": "camera_motion_compensated_tracking",
            "median_frame_detections": median_detections,
            "max_frame_detections": maximum_detections,
            "tracked_objects": count,
            "scene_tracks": len(scene_tracks),
            "camera_registration_successes": registration_successes,
            "camera_registration_failures": registration_failures,
            "processed_frames": processed_frames,
            "frame_stride": frame_stride,
            "duration_seconds": round(duration, 2),
        }
    except HTTPException:
        raise
    except Exception as error:
        request_id = request.state.request_id
        logger.exception("video_count_failed request_id=%s filename=%s error=%s", request_id, file.filename, error)
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed. Check backend logs for request {request_id}.",
        ) from error
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/")
async def health_check():
    return {"status": "ok"}
