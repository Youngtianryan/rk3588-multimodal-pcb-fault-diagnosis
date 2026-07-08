#!/usr/bin/env python3
import json
import math
import os
import socket
import signal
import subprocess
import threading
import time
import re
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from itertools import product
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
HOST = "0.0.0.0"
PORT = 8765
PROFILE_PATH = ROOT / "fusion_profile.json"
DEFAULT_BOARD_MODEL_PATH = ROOT / "board_model.json"
RUNTIME_BOARD_MODEL_PATH = ROOT / "runtime_board_model.json"
MANUAL_BOARD_POSE_PATH = ROOT / "manual_board_pose.json"
MANUAL_THERMAL_POSE_PATH = ROOT / "manual_thermal_pose.json"
YOLO_OBB_RESULT_PATH = ROOT / "yolo_obb_result.json"
REPORTS_DIR = ROOT / "reports"
LLM_CONFIG_PATH = ROOT / "llm_config.json"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/chat/completions"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
LOCAL_LLM_ROOT_CANDIDATES = [
    Path(os.environ.get("RKLLM_ROOT") or "").expanduser() if os.environ.get("RKLLM_ROOT") else None,
    Path("/home/elf/deepseek/rknn_run"),
    Path("/home/deepseek/rknn_run"),
]
LOCAL_LLM_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
LOCAL_LLM_MAX_NEW_TOKENS = 96
LOCAL_LLM_MAX_CONTEXT_LEN = 96
LOCAL_LLM_TIMEOUT = 180

BOARD_MODEL = {
    "id": "study-pcb-2026-06-24",
    "widthMm": 80.0,
    "heightMm": 50.0,
    "components": [],
    "labelRefs": [],
    "holeCentersMm": {},
    "sourceFile": None,
    "isRuntime": False,
}
BOARD_MODEL["aspect"] = BOARD_MODEL["widthMm"] / BOARD_MODEL["heightMm"]
OBB_DISPLAY_THRESHOLD = 0.60

VISIBLE_DEVICE = "/dev/v4l/by-id/usb-BC-260326-ZW_MF500_camera_01.00.00-video-index0"
THERMAL_DEVICE = "/dev/v4l/by-id/usb-HIK_HikCamera_E79351707-video-index0"


def resolve_local_llm_paths():
    checked = []
    for candidate in LOCAL_LLM_ROOT_CANDIDATES:
        if not candidate:
            continue
        root = Path(candidate)
        checked.append(str(root))
        binary = root / "llm_demo"
        model = root / LOCAL_LLM_MODEL_NAME
        if binary.exists() and model.exists():
            return root, binary, model, checked
    fallback_root = next((Path(item) for item in LOCAL_LLM_ROOT_CANDIDATES if item), Path("/home/elf/deepseek/rknn_run"))
    return fallback_root, fallback_root / "llm_demo", fallback_root / LOCAL_LLM_MODEL_NAME, checked


def normalize_layer_name(value):
    text = str(value or "").strip().upper()
    compact = re.sub(r"[\s_\-./]+", "", text)
    if compact in {"", "T", "TOP", "TOPLAYER", "TOPSIDE", "FRONT", "COMPONENT", "COMPONENTSIDE"}:
        return "T"
    if compact in {"B", "BOT", "BOTTOM", "BOTTOMLAYER", "BOTTOMSIDE", "BACK", "SOLDER", "SOLDERSIDE"}:
        return "B"
    return text or "T"


def is_visible_side_component(component):
    return normalize_layer_name(component.get("layer")) != "B"


def component_label_board_point(component):
    ref_x = parse_float_maybe(component.get("refX"))
    ref_y = parse_float_maybe(component.get("refY"))
    if ref_x is not None and ref_y is not None:
        return np.asarray([float(ref_x), float(ref_y)], dtype=np.float32)
    return np.asarray([float(component["x"]), float(component["y"])], dtype=np.float32)


def normalize_board_model(model):
    model = dict(model or {})
    merged = dict(BOARD_MODEL)
    merged.update(model)
    merged["widthMm"] = float(merged.get("widthMm", 80.0))
    merged["heightMm"] = float(merged.get("heightMm", 50.0))
    merged["aspect"] = merged["widthMm"] / max(merged["heightMm"], 1e-6)
    if not isinstance(merged.get("components"), list):
        merged["components"] = []
    if not isinstance(merged.get("labelRefs"), list):
        merged["labelRefs"] = []
    if not isinstance(merged.get("holeCentersMm"), dict):
        merged["holeCentersMm"] = {}
    if not isinstance(merged.get("nets"), dict):
        merged["nets"] = {}
    normalized_components = []
    for item in merged.get("components", []):
        if not isinstance(item, dict):
            continue
        component = dict(item)
        component["layer"] = normalize_layer_name(component.get("layer"))
        normalized_components.append(component)
    merged["components"] = normalized_components
    visible_refs = sorted(
        {
            str(item.get("ref", "")).strip()
            for item in normalized_components
            if str(item.get("ref", "")).strip() and is_visible_side_component(item)
        }
    )
    current_label_refs = [str(ref).strip() for ref in merged.get("labelRefs", []) if str(ref).strip()]
    if visible_refs and (not current_label_refs or (merged.get("sourceFile") and len(current_label_refs) < len(visible_refs))):
        merged["labelRefs"] = visible_refs
    merged["id"] = str(merged.get("id") or "runtime-board").strip()
    merged["sourceFile"] = merged.get("sourceFile")
    merged["isRuntime"] = bool(merged.get("isRuntime", False))
    return merged


def load_board_model_from_path(path):
    model = dict(BOARD_MODEL)
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            model.update(payload)
    except Exception:
        pass
    return normalize_board_model(model)


def load_board_model():
    base = load_board_model_from_path(DEFAULT_BOARD_MODEL_PATH)
    if RUNTIME_BOARD_MODEL_PATH.exists():
        runtime = load_board_model_from_path(RUNTIME_BOARD_MODEL_PATH)
        runtime["isRuntime"] = True
        return normalize_board_model(runtime)
    return normalize_board_model(base)


def save_runtime_board_model(model):
    payload = normalize_board_model(model)
    payload["isRuntime"] = True
    RUNTIME_BOARD_MODEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def clear_runtime_board_model():
    try:
        if RUNTIME_BOARD_MODEL_PATH.exists():
            RUNTIME_BOARD_MODEL_PATH.unlink()
    except Exception:
        pass


def set_active_board_model(model, persist=True):
    global BOARD_MODEL
    payload = normalize_board_model(model)
    payload["isRuntime"] = bool(persist)
    BOARD_MODEL = payload
    try:
        COMPONENT_PROJECTION_CACHE["signature"] = None
        COMPONENT_PROJECTION_CACHE["entries"] = None
    except NameError:
        pass
    if persist:
        save_runtime_board_model(payload)
    return BOARD_MODEL


def parse_float_maybe(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text.replace(",", ""), re.IGNORECASE)
        if not match:
            return None
        return float(match.group(0))
    except Exception:
        return None


def infer_board_geometry_from_components(components):
    points = []
    for component in components:
        x = parse_float_maybe(component.get("x"))
        y = parse_float_maybe(component.get("y"))
        if x is None or y is None:
            continue
        points.append((x, y))
    if not points:
        return 80.0, 50.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return max(width + 4.0, 10.0), max(height + 4.0, 10.0)


def infer_hole_centers_from_components(components, width_mm, height_mm):
    holes = []
    for component in components:
        ref = str(component.get("ref", "")).strip()
        footprint = str(component.get("footprint", "")).lower()
        comment = str(component.get("comment", "")).lower()
        if "hole" not in ref.lower() and "m3" not in footprint and "hole" not in footprint and "mount" not in comment:
            continue
        x = parse_float_maybe(component.get("x"))
        y = parse_float_maybe(component.get("y"))
        if x is None or y is None:
            continue
        holes.append({"ref": ref or f"hole{len(holes)+1}", "x": x, "y": y})

    if len(holes) < 4:
        return {}

    ordered = sorted(holes, key=lambda item: (item["y"], item["x"]))
    bottom = sorted(ordered[:2], key=lambda item: item["x"])
    top = sorted(ordered[-2:], key=lambda item: item["x"])
    return {
        "hole3": {"x": top[0]["x"], "y": top[0]["y"]},
        "hole1": {"x": top[1]["x"], "y": top[1]["y"]},
        "hole4": {"x": bottom[0]["x"], "y": bottom[0]["y"]},
        "hole2": {"x": bottom[1]["x"], "y": bottom[1]["y"]},
    }


def build_runtime_board_model(upload_payload):
    file_name = str(upload_payload.get("fileName") or "uploaded-board").strip()
    source_components = upload_payload.get("components") or []
    components = []
    for item in source_components:
        ref = str(item.get("ref", "")).strip()
        x = parse_float_maybe(item.get("x"))
        y = parse_float_maybe(item.get("y"))
        if not ref or x is None or y is None:
            continue
        component = {
            "ref": ref,
            "x": x,
            "y": y,
            "layer": normalize_layer_name(item.get("layer", "T")),
            "rotation": float(parse_float_maybe(item.get("rotation")) or 0.0),
            "footprint": str(item.get("footprint", "") or "").strip(),
            "device": str(item.get("device", "") or "").strip(),
            "comment": str(item.get("comment", "") or "").strip(),
            "name": str(item.get("name", "") or "").strip(),
            "pins": int(parse_float_maybe(item.get("pins")) or 0),
            "smd": str(item.get("smd", "") or "").strip(),
        }
        ref_x = parse_float_maybe(item.get("refX"))
        ref_y = parse_float_maybe(item.get("refY"))
        pad_x = parse_float_maybe(item.get("padX"))
        pad_y = parse_float_maybe(item.get("padY"))
        if ref_x is not None and ref_y is not None:
            component["refX"] = ref_x
            component["refY"] = ref_y
        if pad_x is not None and pad_y is not None:
            component["padX"] = pad_x
            component["padY"] = pad_y
        components.append(component)
    if not components:
        raise ValueError(
            "no component coordinates parsed; expected columns such as Designator/Ref, Mid X/X, Mid Y/Y"
        )

    width_mm = parse_float_maybe(upload_payload.get("widthMm"))
    height_mm = parse_float_maybe(upload_payload.get("heightMm"))
    if width_mm is None or height_mm is None:
        inferred_w, inferred_h = infer_board_geometry_from_components(components)
        width_mm = width_mm or inferred_w
        height_mm = height_mm or inferred_h

    hole_centers = upload_payload.get("holeCentersMm")
    if not isinstance(hole_centers, dict) or len(hole_centers) < 4:
        hole_centers = infer_hole_centers_from_components(components, width_mm, height_mm)

    label_refs = sorted(
        {
            str(item.get("ref", "")).strip()
            for item in components
            if str(item.get("ref", "")).strip() and is_visible_side_component(item)
        }
    )
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(file_name).stem).strip("-").lower() or "uploaded-board"
    model = {
        "id": safe_id,
        "widthMm": width_mm,
        "heightMm": height_mm,
        "components": components,
        "labelRefs": label_refs,
        "holeCentersMm": hole_centers,
        "nets": upload_payload.get("nets") if isinstance(upload_payload.get("nets"), dict) else {},
        "sourceFile": file_name,
    }
    return normalize_board_model(model)


BOARD_MODEL = load_board_model()


def load_manual_board_pose():
    try:
        if not MANUAL_BOARD_POSE_PATH.exists():
            return None
        payload = json.loads(MANUAL_BOARD_POSE_PATH.read_text(encoding="utf-8"))
        if str(payload.get("boardId") or "").strip() != str(BOARD_MODEL.get("id") or "").strip():
            return None
        points = np.asarray(payload.get("imagePoints"), dtype=np.float32)
        if points.shape != (4, 2):
            return None
        return {
            "imagePoints": points.astype(np.float32),
            "savedAt": float(payload.get("savedAt", time.time())),
        }
    except Exception:
        return None


def save_manual_board_pose(image_points):
    points = np.asarray(image_points, dtype=np.float32)
    payload = {
        "boardId": BOARD_MODEL["id"],
        "savedAt": time.time(),
        "imagePoints": np.round(points, 3).tolist(),
        "pointOrder": ["hole3-top-left", "hole1-top-right", "hole2-bottom-right", "hole4-bottom-left"],
    }
    MANUAL_BOARD_POSE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manual_thermal_pose():
    try:
        if not MANUAL_THERMAL_POSE_PATH.exists():
            return None
        payload = json.loads(MANUAL_THERMAL_POSE_PATH.read_text(encoding="utf-8"))
        if str(payload.get("boardId") or "").strip() != str(BOARD_MODEL.get("id") or "").strip():
            return None
        points = np.asarray(payload.get("imagePoints"), dtype=np.float32)
        if points.shape != (4, 2):
            return None
        return {
            "imagePoints": points.astype(np.float32),
            "savedAt": float(payload.get("savedAt", time.time())),
        }
    except Exception:
        return None


def save_manual_thermal_pose(image_points):
    points = np.asarray(image_points, dtype=np.float32)
    payload = {
        "boardId": BOARD_MODEL["id"],
        "savedAt": time.time(),
        "imagePoints": np.round(points, 3).tolist(),
        "pointOrder": ["hole3-top-left", "hole1-top-right", "hole2-bottom-right", "hole4-bottom-left"],
    }
    MANUAL_THERMAL_POSE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_yolo_obb_result():
    try:
        if not YOLO_OBB_RESULT_PATH.exists():
            return {
                "enabled": False,
                "running": False,
                "available": False,
                "message": "not-run",
                "savedAt": None,
                "modelPath": None,
                "count": 0,
                "detections": [],
            }
        payload = json.loads(YOLO_OBB_RESULT_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid payload")
        payload.setdefault("enabled", False)
        payload.setdefault("running", False)
        payload.setdefault("available", False)
        payload.setdefault("message", "loaded")
        payload.setdefault("savedAt", None)
        payload.setdefault("modelPath", None)
        payload.setdefault("detections", [])
        payload["detections"] = [det for det in payload["detections"] if float(det.get("score", 0.0) or 0.0) >= OBB_DISPLAY_THRESHOLD]
        payload["count"] = len(payload["detections"])
        return payload
    except Exception:
        return {
            "enabled": False,
            "running": False,
            "available": False,
            "message": "load-failed",
            "savedAt": None,
            "modelPath": None,
            "count": 0,
            "detections": [],
        }


def save_yolo_obb_result(payload):
    payload = dict(payload)
    payload["detections"] = [det for det in payload.get("detections", []) if float(det.get("score", 0.0) or 0.0) >= OBB_DISPLAY_THRESHOLD]
    payload["count"] = len(payload["detections"])
    YOLO_OBB_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_manual_pose_files():
    for path in (MANUAL_BOARD_POSE_PATH, MANUAL_THERMAL_POSE_PATH):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def http_post_json(url, payload, timeout=6):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset("utf-8")
        body = response.read().decode(charset)
    return json.loads(body) if body else {}


def http_post_json_headers(url, payload, headers=None, timeout=12):
    data = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=merged_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset("utf-8")
        body = response.read().decode(charset)
    return json.loads(body) if body else {}


def reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def load_llm_config():
    config = {}
    try:
        if LLM_CONFIG_PATH.exists():
            payload = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                config = payload
    except Exception:
        config = {}
    api_key = str(os.environ.get("DEEPSEEK_API_KEY") or config.get("deepseekApiKey") or "").strip()
    model = str(os.environ.get("DEEPSEEK_MODEL") or config.get("model") or DEEPSEEK_DEFAULT_MODEL).strip()
    base_url = str(os.environ.get("DEEPSEEK_BASE_URL") or config.get("baseUrl") or DEEPSEEK_BASE_URL).strip().rstrip("/")
    enabled = bool(api_key)
    return {
        "enabled": enabled,
        "provider": "deepseek",
        "apiKey": api_key,
        "model": model or DEEPSEEK_DEFAULT_MODEL,
        "baseUrl": base_url or DEEPSEEK_BASE_URL,
    }


def local_llm_config():
    file_config = {}
    try:
        if LLM_CONFIG_PATH.exists():
            payload = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                file_config = payload
    except Exception:
        file_config = {}
    def clamp_int(*values, default, minimum, maximum):
        for value in values:
            try:
                parsed = int(str(value).strip())
            except Exception:
                continue
            return max(minimum, min(parsed, maximum))
        return default
    root, binary, model, checked_roots = resolve_local_llm_paths()
    server_url = str(os.environ.get("RKLLM_SERVER_URL") or file_config.get("localServerUrl") or "").strip().rstrip("/")
    server_model = str(os.environ.get("RKLLM_SERVER_MODEL") or file_config.get("localServerModel") or "rkllm").strip()
    max_new_tokens = clamp_int(
        os.environ.get("RKLLM_MAX_NEW_TOKENS"),
        file_config.get("localMaxNewTokens"),
        default=LOCAL_LLM_MAX_NEW_TOKENS,
        minimum=32,
        maximum=512,
    )
    max_context_len = clamp_int(
        os.environ.get("RKLLM_MAX_CONTEXT_LEN"),
        file_config.get("localMaxContextLen"),
        default=LOCAL_LLM_MAX_CONTEXT_LEN,
        minimum=64,
        maximum=512,
    )
    return {
        "enabled": bool(server_url) or (binary.exists() and model.exists()),
        "provider": "deepseek-local",
        "serverUrl": server_url,
        "serverModel": server_model or "rkllm",
        "rootPath": str(root),
        "binaryPath": str(binary),
        "modelPath": str(model),
        "checkedRoots": checked_roots,
        "maxNewTokens": max_new_tokens,
        "maxContextLen": max_context_len,
    }


def normalize_llm_mode(value):
    mode = str(value or "auto").strip().lower()
    if mode in ("api", "local", "auto"):
        return mode
    return "auto"


def llm_mode_label(mode):
    labels = {
        "auto": "自动",
        "api": "DeepSeek API",
        "local": "本地 DeepSeek",
    }
    return labels.get(normalize_llm_mode(mode), "自动")


def network_endpoint_reachable(url, timeout=2.0):
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=float(timeout)):
            return True
    except Exception:
        return False


def default_manual_adjust():
    return {
        "dx": 0.0,
        "dy": 0.0,
        "scaleX": 1.0,
        "scaleY": 1.0,
        "sourceLeft": 0.0,
        "sourceRight": 0.0,
        "sourceTop": 0.0,
        "sourceBottom": 0.0,
    }


def serialize_matrix(matrix):
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float32).round(8).tolist()


def deserialize_matrix(payload):
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float32)
    if matrix.shape != (3, 3):
        return None
    return matrix


def normalize_manual_adjust(adjust):
    merged = default_manual_adjust()
    if adjust:
        for key in merged:
            if key in adjust:
                merged[key] = float(adjust[key])
    merged["dx"] = float(np.clip(merged["dx"], -1200, 1200))
    merged["dy"] = float(np.clip(merged["dy"], -1200, 1200))
    merged["scaleX"] = float(np.clip(merged["scaleX"], 0.35, 2.5))
    merged["scaleY"] = float(np.clip(merged["scaleY"], 0.35, 2.5))
    for key in ("sourceLeft", "sourceRight", "sourceTop", "sourceBottom"):
        merged[key] = float(np.clip(merged[key], -120, 120))
    return merged


def load_profile_store():
    store = {
        "profiles": [],
        "lastManualAdjust": default_manual_adjust(),
        "rigMatrix": None,
        "rigCalibration": None,
    }
    try:
        if not PROFILE_PATH.exists():
            return store
        payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload.get("profiles"), list):
            store["profiles"] = payload["profiles"]
        if "manualAdjust" in payload:
            store["lastManualAdjust"] = normalize_manual_adjust(payload.get("manualAdjust"))
        elif "lastManualAdjust" in payload:
            store["lastManualAdjust"] = normalize_manual_adjust(payload.get("lastManualAdjust"))
        store["rigMatrix"] = deserialize_matrix(payload.get("rigMatrix"))
        if isinstance(payload.get("rigCalibration"), dict):
            store["rigCalibration"] = payload.get("rigCalibration")
    except Exception:
        return store
    return store


def save_profile_store(store):
    payload = {
        "savedAt": time.time(),
        "lastManualAdjust": normalize_manual_adjust(store.get("lastManualAdjust")),
        "profiles": store.get("profiles", []),
        "rigMatrix": serialize_matrix(store.get("rigMatrix")),
        "rigCalibration": store.get("rigCalibration"),
    }
    PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fingerprint_from_quads(visible_quad, thermal_quad, visible_shape, thermal_shape):
    return {
        "visibleAspect": round(float(quad_aspect(visible_quad)), 4),
        "visibleArea": round(float(abs(cv2.contourArea(visible_quad)) / max(1.0, visible_shape[0] * visible_shape[1])), 6),
        "thermalAspect": round(float(quad_aspect(thermal_quad)), 4),
        "thermalArea": round(float(abs(cv2.contourArea(thermal_quad)) / max(1.0, thermal_shape[0] * thermal_shape[1])), 6),
    }


def fingerprint_score(current, saved):
    return (
        abs(current["visibleAspect"] - saved.get("visibleAspect", current["visibleAspect"])) * 3.0
        + abs(current["visibleArea"] - saved.get("visibleArea", current["visibleArea"])) * 20.0
        + abs(current["thermalAspect"] - saved.get("thermalAspect", current["thermalAspect"])) * 2.0
        + abs(current["thermalArea"] - saved.get("thermalArea", current["thermalArea"])) * 20.0
    )


def select_profile_adjust(fingerprint):
    store = load_profile_store()
    best = None
    best_score = 1e9
    for profile in store["profiles"]:
        profile_fingerprint = profile.get("fingerprint")
        if not isinstance(profile_fingerprint, dict):
            continue
        score = fingerprint_score(fingerprint, profile_fingerprint)
        if score < best_score:
            best_score = score
            best = profile

    if best is not None and best_score < 1.0:
        return normalize_manual_adjust(best.get("manualAdjust")), {
            "profileId": best.get("id"),
            "score": round(float(best_score), 4),
            "matched": True,
        }

    if not store["profiles"]:
        return normalize_manual_adjust(store.get("lastManualAdjust")), {
            "profileId": "legacy-last-manual",
            "score": None,
            "matched": False,
        }

    return default_manual_adjust(), {"profileId": None, "score": None, "matched": False}


def save_profile_adjust(adjust):
    store = load_profile_store()
    store["lastManualAdjust"] = normalize_manual_adjust(adjust)
    save_profile_store(store)


def load_rig_calibration():
    store = load_profile_store()
    return store.get("rigMatrix"), store.get("rigCalibration")


def save_rig_calibration(matrix, visible_quad=None, thermal_quad=None):
    store = load_profile_store()
    store["rigMatrix"] = np.asarray(matrix, dtype=np.float32)
    store["rigCalibration"] = {
        "savedAt": time.time(),
        "boardId": BOARD_MODEL["id"],
        "visibleQuad": None if visible_quad is None else np.round(np.asarray(visible_quad), 2).tolist(),
        "thermalQuad": None if thermal_quad is None else np.round(np.asarray(thermal_quad), 2).tolist(),
    }
    save_profile_store(store)


def save_profile_for_board(adjust, fingerprint):
    store = load_profile_store()
    manual = normalize_manual_adjust(adjust)
    store["lastManualAdjust"] = manual

    best_index = None
    best_score = 1e9
    for index, profile in enumerate(store["profiles"]):
        profile_fingerprint = profile.get("fingerprint")
        if not isinstance(profile_fingerprint, dict):
            continue
        score = fingerprint_score(fingerprint, profile_fingerprint)
        if score < best_score:
            best_score = score
            best_index = index

    profile_payload = {
        "id": f"board-{int(time.time())}",
        "savedAt": time.time(),
        "fingerprint": fingerprint,
        "manualAdjust": manual,
    }

    if best_index is not None and best_score < 1.0:
        profile_payload["id"] = store["profiles"][best_index].get("id", profile_payload["id"])
        store["profiles"][best_index] = profile_payload
    else:
        store["profiles"].append(profile_payload)

    save_profile_store(store)


class CameraWorker:
    def __init__(
        self,
        name,
        device,
        width,
        height,
        rotate=None,
        jpeg_quality=88,
        preview_max_width=None,
        sharpen_preview=False,
        stream_interval=0.08,
    ):
        self.name = name
        self.device = os.path.realpath(device)
        self.width = width
        self.height = height
        self.rotate = rotate
        self.jpeg_quality = jpeg_quality
        self.preview_max_width = None if preview_max_width is None else int(max(320, preview_max_width))
        self.sharpen_preview = bool(sharpen_preview)
        self.stream_interval = max(0.03, float(stream_interval))
        self.capture = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_jpeg = None
        self.latest_debug_jpeg = None
        self.latest_shape = None
        self.fps = 0.0
        self.online = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _transform(self, frame):
        if self.rotate == "clockwise":
          return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotate == "counterclockwise":
          return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.rotate == "180":
          return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _prepare_preview_frame(self, frame):
        if frame is None or self.preview_max_width is None:
            return frame
        height, width = frame.shape[:2]
        if width <= self.preview_max_width:
            preview = frame
        else:
            scale = self.preview_max_width / max(float(width), 1.0)
            preview = cv2.resize(
                frame,
                (self.preview_max_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        if self.sharpen_preview:
            preview = cv2.addWeighted(preview, 1.10, cv2.GaussianBlur(preview, (0, 0), 0.8), -0.10, 0)
        return preview

    def _run(self):
        frames = 0
        started = time.time()

        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.capture = self._open()
                if not self.capture.isOpened():
                    self.online = False
                    time.sleep(1.0)
                    continue

            try:
                ok, frame = self.capture.read()
            except cv2.error:
                ok = False
                frame = None
            if not ok or frame is None:
                self.online = False
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None
                time.sleep(0.5)
                continue

            frame = self._transform(frame)
            preview_frame = self._prepare_preview_frame(frame)
            ok, encoded = cv2.imencode(
                ".jpg",
                preview_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue

            frames += 1
            elapsed = max(time.time() - started, 1e-6)
            self.fps = frames / elapsed
            self.online = True

            with self.lock:
                self.latest_frame = frame
                self.latest_shape = frame.shape[:2]
                self.latest_jpeg = encoded.tobytes()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_debug_jpeg(self):
        with self.lock:
            return self.latest_debug_jpeg

    def get_frame(self, copy_frame=True):
        with self.lock:
            if self.latest_frame is None:
                return None
            if copy_frame:
                return self.latest_frame.copy()
            return self.latest_frame

    def get_status(self):
        with self.lock:
            if self.latest_shape is None:
                width = self.width
                height = self.height
            else:
                height, width = self.latest_shape
            return {
                "name": self.name,
                "online": self.online,
                "device": self.device,
                "width": int(width),
                "height": int(height),
                "fps": float(self.fps),
            }


visible_worker = CameraWorker(
    "visible",
    VISIBLE_DEVICE,
    2560,
    1440,
    rotate=None,
    jpeg_quality=76,
    preview_max_width=1280,
    sharpen_preview=True,
    stream_interval=0.066,
)
thermal_worker = CameraWorker(
    "thermal",
    THERMAL_DEVICE,
    240,
    320,
    rotate="clockwise",
    jpeg_quality=82,
    preview_max_width=480,
    stream_interval=0.066,
)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class FusionWorker:
    def __init__(self, visible, thermal, alpha=0.42, jpeg_quality=74, preview_max_width=1120, frame_interval=0.066):
        self.visible = visible
        self.thermal = thermal
        self.alpha = alpha
        self.jpeg_quality = jpeg_quality
        self.preview_max_width = int(max(960, preview_max_width))
        self.frame_interval = max(0.03, float(frame_interval))
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_debug_jpeg = None
        self.latest_status = {
            "online": False,
            "mode": "waiting",
            "calibrated": False,
            "visibleBorder": False,
            "samples": 0,
            "fps": 0.0,
        }
        self.calibration_lock = threading.Lock()
        self.matrix = None
        self.visible_quad = None
        self.thermal_quad = None
        self.thermal_rotation = "capture-cw"
        self.board_matrix = None
        self.thermal_board_matrix = None
        self.board_pose_method = "none"
        self.hole_debug = {"method": "none", "accepted": False, "candidates": []}
        self.manual_board_pose = load_manual_board_pose()
        self.manual_thermal_pose = load_manual_thermal_pose()
        self.manual_adjust = default_manual_adjust()
        self.profile_match = {"profileId": None, "score": None, "matched": False}
        self.board_fingerprint = None
        self.hotspot = None
        self.hotspot_analysis = None
        self.cached_thermal_ocr = None
        self.hotspot_last_refresh = 0.0
        self.hotspot_last_ocr_refresh = 0.0
        self.hotspot_refresh_interval = 0.22
        self.hotspot_ocr_refresh_interval = 1.2
        self.hotspot_worker_interval = 1.15
        self.debug_jpeg_last_refresh = 0.0
        self.debug_jpeg_refresh_interval = 0.55
        self.debug_jpeg_signature = None
        self.board_overlay_signature = None
        self.board_overlay_image = None
        self.obb_overlay_signature = None
        self.obb_overlay_image = None
        self.report_state_lock = threading.Lock()
        self.report_state = {
            "running": False,
            "progress": 0,
            "stage": "idle",
            "message": "idle",
            "requestedMode": "auto",
            "requestedModeLabel": "自动",
            "provider": None,
            "providerLabel": None,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
            "pdfPath": None,
            "pdfUrl": None,
        }
        self.calibration_notice = None
        self.manual_pose_validation_last_check = 0.0
        self.manual_pose_validation_interval = 1.2
        self.samples = 0
        self.yolo_obb = load_yolo_obb_result()
        self.display_annotations = False
        self.recalibrate_requested = True
        self.rig_matrix, self.rig_calibration = load_rig_calibration()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.hotspot_thread = threading.Thread(target=self._run_hotspot_worker, daemon=True)

    def start(self):
        self.thread.start()
        self.hotspot_thread.start()

    def stop(self):
        self.running = False

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_debug_jpeg(self):
        with self.lock:
            return self.latest_debug_jpeg

    def get_status(self):
        with self.lock:
            return dict(self.latest_status)

    def get_report_state(self):
        with self.report_state_lock:
            return deepcopy(self.report_state)

    def _set_report_state(self, **updates):
        with self.report_state_lock:
            self.report_state.update(updates)

    def _reset_report_state(self):
        with self.report_state_lock:
            self.report_state = {
                "running": False,
                "progress": 0,
                "stage": "idle",
                "message": "idle",
                "requestedMode": "auto",
                "requestedModeLabel": "自动",
                "provider": None,
                "providerLabel": None,
                "startedAt": None,
                "finishedAt": None,
                "error": None,
                "pdfPath": None,
                "pdfUrl": None,
            }

    def start_diagnostic_report(self, llm_api_url=None, llm_mode="auto"):
        llm_mode = normalize_llm_mode(llm_mode)
        with self.report_state_lock:
            if self.report_state.get("running"):
                return {"ok": False, "error": "report generation already running", "task": deepcopy(self.report_state)}
            self.report_state = {
                "running": True,
                "progress": 1,
                "stage": "queued",
                "message": "报告任务已启动",
                "provider": None,
                "providerLabel": None,
                "startedAt": time.time(),
                "finishedAt": None,
                "error": None,
                "pdfPath": None,
                "pdfUrl": None,
            }
            self.report_state["requestedMode"] = llm_mode
            self.report_state["requestedModeLabel"] = llm_mode_label(llm_mode)
            task = deepcopy(self.report_state)

        def worker():
            self.generate_diagnostic_report(llm_api_url=llm_api_url, llm_mode=llm_mode)

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "started": True, "task": task}

    def get_yolo_obb_status(self):
        with self.calibration_lock:
            return summarize_yolo_obb_state(deepcopy(self.yolo_obb))

    def get_settings(self):
        with self.calibration_lock:
            return {
                "alpha": float(self.alpha),
                "manualAdjust": dict(self.manual_adjust),
                "profileMatch": dict(self.profile_match),
            }

    def _clear_hotspot_runtime_cache(self):
        self.hotspot = None
        self.hotspot_analysis = None
        self.cached_thermal_ocr = None
        self.hotspot_last_refresh = 0.0
        self.hotspot_last_ocr_refresh = 0.0

    def _run_hotspot_worker(self):
        while self.running:
            started = time.time()
            try:
                visible = self.visible.get_frame(copy_frame=False)
                thermal = self.thermal.get_frame(copy_frame=False)
                if visible is None or thermal is None:
                    time.sleep(0.25)
                    continue

                with self.calibration_lock:
                    matrix = None if self.matrix is None else np.asarray(self.matrix, dtype=np.float32).copy()
                    rig_matrix = None if self.rig_matrix is None else np.asarray(self.rig_matrix, dtype=np.float32).copy()
                    visible_quad = None if self.visible_quad is None else np.asarray(self.visible_quad, dtype=np.float32).copy()
                    thermal_quad = None if self.thermal_quad is None else np.asarray(self.thermal_quad, dtype=np.float32).copy()
                    board_matrix = None if self.board_matrix is None else np.asarray(self.board_matrix, dtype=np.float32).copy()
                    manual_adjust = dict(self.manual_adjust)
                    yolo_detections = deepcopy(self.yolo_obb.get("detections", []))

                if visible_quad is None or (matrix is None and rig_matrix is None):
                    time.sleep(0.2)
                    continue

                analysis = analyze_hotspot_from_frames(
                    visible,
                    thermal,
                    matrix=matrix,
                    rig_matrix=rig_matrix,
                    visible_quad=visible_quad,
                    thermal_quad=thermal_quad,
                    board_matrix=board_matrix,
                    manual_adjust=manual_adjust,
                    yolo_detections=yolo_detections,
                )
                with self.calibration_lock:
                    self.hotspot_analysis = deepcopy(analysis)
                    self.hotspot = None if analysis is None else analysis.get("imagePoint")
            except Exception:
                pass

            elapsed = time.time() - started
            time.sleep(max(0.08, self.hotspot_worker_interval - elapsed))

    def _invalidate_debug_frame_cache(self):
        self.debug_jpeg_last_refresh = 0.0
        self.debug_jpeg_signature = None

    def _invalidate_overlay_cache(self):
        self.board_overlay_signature = None
        self.board_overlay_image = None
        self.obb_overlay_signature = None
        self.obb_overlay_image = None

    def _clear_manual_pose_state_unlocked(self, reason=None):
        clear_manual_pose_files()
        self.manual_board_pose = None
        self.manual_thermal_pose = None
        self.matrix = None
        self.visible_quad = None
        self.thermal_quad = None
        self.thermal_board_matrix = None
        self.thermal_rotation = "capture-cw"
        self.board_matrix = None
        self.board_pose_method = "none"
        self.hole_debug = {"method": "none", "accepted": False, "candidates": []}
        self.profile_match = {"profileId": BOARD_MODEL.get("id"), "score": None, "matched": False}
        self.board_fingerprint = None
        self.samples = 0
        self.recalibrate_requested = True
        self.calibration_notice = reason
        self._clear_hotspot_runtime_cache()
        self._invalidate_debug_frame_cache()
        self._invalidate_overlay_cache()

    def _manual_pose_motion_threshold(self, quad):
        quad = np.asarray(quad, dtype=np.float32)
        edges = [
            float(np.linalg.norm(quad[0] - quad[1])),
            float(np.linalg.norm(quad[1] - quad[2])),
            float(np.linalg.norm(quad[2] - quad[3])),
            float(np.linalg.norm(quad[3] - quad[0])),
        ]
        long_edge = max(edges) if edges else 0.0
        return max(26.0, long_edge * 0.045)

    def _invalidate_stale_manual_pose_if_needed(self, visible):
        now = time.time()
        if (now - self.manual_pose_validation_last_check) < self.manual_pose_validation_interval:
            return
        self.manual_pose_validation_last_check = now
        if self.manual_board_pose is not None and self.manual_thermal_pose is not None:
            return
        if visible is None or self.manual_board_pose is None or self.manual_thermal_pose is None:
            return
        scale = 1.0
        working = visible
        if visible.shape[1] > 1280:
            scale = 1280.0 / float(visible.shape[1])
            working = cv2.resize(visible, (int(visible.shape[1] * scale), int(visible.shape[0] * scale)))
        current_quad = find_board_quad(working, prefer_color=True)
        if current_quad is None:
            return
        if scale != 1.0:
            current_quad = current_quad.astype(np.float32) / scale
        manual_points = order_quad(np.asarray(self.manual_board_pose["imagePoints"], dtype=np.float32))
        manual_matrix = board_matrix_from_hole_points(manual_points)
        if manual_matrix is None:
            return
        saved_quad = order_quad(board_points_to_image(manual_matrix, board_corners_mm()).astype(np.float32))
        current_quad = order_quad(np.asarray(current_quad, dtype=np.float32))

        mean_corner_distance = float(np.mean(np.linalg.norm(saved_quad - current_quad, axis=1)))
        center_shift = float(np.linalg.norm(saved_quad.mean(axis=0) - current_quad.mean(axis=0)))
        saved_area = abs(cv2.contourArea(saved_quad.astype(np.float32)))
        current_area = abs(cv2.contourArea(current_quad.astype(np.float32)))
        area_ratio = (current_area / saved_area) if saved_area > 1.0 else 1.0
        threshold = self._manual_pose_motion_threshold(current_quad)

        moved = (
            (mean_corner_distance > threshold and center_shift > threshold * 0.7)
            or area_ratio < 0.72
            or area_ratio > 1.38
        )
        if not moved:
            return

        with self.calibration_lock:
            if self.manual_board_pose is None:
                return
            self._clear_manual_pose_state_unlocked(
                reason=f"board-moved(mean={mean_corner_distance:.1f},center={center_shift:.1f},scale={area_ratio:.2f})"
            )

    def _build_debug_frame_signature(self):
        visible_quad = None if self.visible_quad is None else tuple(np.round(np.asarray(self.visible_quad, dtype=np.float32).reshape(-1), 1).tolist())
        board_matrix = None if self.board_matrix is None else tuple(np.round(np.asarray(self.board_matrix, dtype=np.float32).reshape(-1), 3).tolist())
        hole_points = []
        for item in (self.hole_debug or {}).get("candidates", []):
            point = ((item.get("best") or {}).get("point")) or []
            if len(point) == 2:
                hole_points.append((round(float(point[0]), 1), round(float(point[1]), 1)))
        return (
            self.board_pose_method,
            visible_quad,
            board_matrix,
            tuple(hole_points),
        )

    def _get_cached_hotspot_analysis(self, thermal, adjusted_matrix, board_matrix, visible_quad, thermal_quad, manual_adjust, yolo_detections, visible_shape):
        now = time.time()
        existing = deepcopy(self.hotspot_analysis) if self.hotspot_analysis is not None else None
        if existing is not None and (now - self.hotspot_last_refresh) < self.hotspot_refresh_interval:
            return existing

        need_ocr = (
            self.cached_thermal_ocr is None
            or existing is None
            or (now - self.hotspot_last_ocr_refresh) >= self.hotspot_ocr_refresh_interval
        )
        analysis = analyze_hotspot(
            thermal,
            adjusted_matrix,
            board_matrix=board_matrix,
            visible_quad=visible_quad,
            thermal_quad=thermal_quad,
            manual_adjust=manual_adjust,
            yolo_detections=yolo_detections,
            visible_shape=visible_shape,
            enable_ocr=need_ocr,
            cached_thermal_ocr=None if need_ocr else deepcopy(self.cached_thermal_ocr),
        )
        if analysis is None:
            return existing

        self.hotspot_analysis = analysis
        self.hotspot_last_refresh = now
        self.hotspot = analysis.get("imagePoint")
        thermal_ocr = analysis.get("thermalOcr")
        if thermal_ocr and thermal_ocr.get("ok"):
            self.cached_thermal_ocr = deepcopy(thermal_ocr)
            if need_ocr:
                self.hotspot_last_ocr_refresh = now
        return deepcopy(analysis)

    def _maybe_refresh_debug_jpeg(self, visible):
        if visible is None:
            return
        now = time.time()
        signature = self._build_debug_frame_signature()
        if (
            self.latest_debug_jpeg is not None
            and signature == self.debug_jpeg_signature
            and (now - self.debug_jpeg_last_refresh) < self.debug_jpeg_refresh_interval
        ):
            return
        self.latest_debug_jpeg = build_hole_debug_frame(visible, self.visible_quad, self.board_matrix, self.hole_debug)
        self.debug_jpeg_signature = signature
        self.debug_jpeg_last_refresh = now

    def _build_board_overlay_signature(self, frame_shape, board_matrix):
        if board_matrix is None:
            return None
        return (
            tuple(frame_shape[:2]),
            BOARD_MODEL.get("id"),
            tuple(np.round(np.asarray(board_matrix, dtype=np.float32).reshape(-1), 3).tolist()),
        )

    def _get_board_overlay(self, frame_shape, board_matrix):
        signature = self._build_board_overlay_signature(frame_shape, board_matrix)
        if signature is None:
            return None
        if self.board_overlay_image is not None and signature == self.board_overlay_signature:
            return self.board_overlay_image
        overlay = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        draw_board_model_overlay(overlay, board_matrix)
        self.board_overlay_signature = signature
        self.board_overlay_image = overlay
        return overlay

    def _build_obb_overlay_signature(self, frame_shape):
        detections = self.yolo_obb.get("detections", [])
        if not self.yolo_obb.get("enabled") or not detections:
            return None
        return (
            tuple(frame_shape[:2]),
            self.yolo_obb.get("savedAt"),
            self.yolo_obb.get("count"),
        )

    def _get_obb_overlay(self, frame_shape, scale_x=1.0, scale_y=1.0):
        signature = self._build_obb_overlay_signature(frame_shape)
        if signature is None:
            return None
        scaled_signature = signature + (round(float(scale_x), 4), round(float(scale_y), 4))
        if self.obb_overlay_image is not None and scaled_signature == self.obb_overlay_signature:
            return self.obb_overlay_image
        overlay = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        detections = []
        for det in self.yolo_obb.get("detections", []):
            item = dict(det)
            quad = item.get("quad") or []
            if len(quad) == 4:
                scaled_quad = []
                for point in quad:
                    if len(point) == 2:
                        scaled_quad.append([float(point[0]) * scale_x, float(point[1]) * scale_y])
                if len(scaled_quad) == 4:
                    item["quad"] = scaled_quad
            center = item.get("center") or []
            if len(center) == 2:
                item["center"] = [float(center[0]) * scale_x, float(center[1]) * scale_y]
            detections.append(item)
        draw_obb_overlay(overlay, detections, threshold=OBB_DISPLAY_THRESHOLD)
        self.obb_overlay_signature = scaled_signature
        self.obb_overlay_image = overlay
        return overlay

    def update_settings(self, alpha=None, manual_adjust=None, reset_manual=False, save_manual=False):
        with self.calibration_lock:
            if alpha is not None:
                self.alpha = float(np.clip(alpha, 0.05, 0.95))
            if reset_manual:
                self.manual_adjust = default_manual_adjust()
            if manual_adjust:
                for key in ("dx", "dy", "scaleX", "scaleY", "sourceLeft", "sourceRight", "sourceTop", "sourceBottom"):
                    if key in manual_adjust:
                        self.manual_adjust[key] = float(manual_adjust[key])
                self.manual_adjust = normalize_manual_adjust(self.manual_adjust)
            if save_manual:
                if self.board_fingerprint is not None:
                    save_profile_for_board(self.manual_adjust, self.board_fingerprint)
                else:
                    save_profile_adjust(self.manual_adjust)
            return {"ok": True, "settings": self.get_settings_unlocked()}

    def get_settings_unlocked(self):
        return {
            "alpha": float(self.alpha),
            "manualAdjust": dict(self.manual_adjust),
            "profileMatch": dict(self.profile_match),
        }

    def analyze_current_hotspot(self, include_frames=False):
        visible = self.visible.get_frame(copy_frame=False)
        thermal = self.thermal.get_frame(copy_frame=False)
        if visible is None or thermal is None:
            return {"ok": False, "error": "visible or thermal frame unavailable"}

        now = time.time()
        with self.calibration_lock:
            matrix = None if self.matrix is None else np.asarray(self.matrix, dtype=np.float32).copy()
            rig_matrix = None if self.rig_matrix is None else np.asarray(self.rig_matrix, dtype=np.float32).copy()
            visible_quad = None if self.visible_quad is None else np.asarray(self.visible_quad, dtype=np.float32).copy()
            thermal_quad = None if self.thermal_quad is None else np.asarray(self.thermal_quad, dtype=np.float32).copy()
            board_matrix = None if self.board_matrix is None else np.asarray(self.board_matrix, dtype=np.float32).copy()
            manual_adjust = dict(self.manual_adjust)
            yolo_detections = deepcopy(self.yolo_obb.get("detections", []))
            cached_analysis = deepcopy(self.hotspot_analysis)
            hotspot_age = now - float(self.hotspot_last_refresh or 0.0)

        if cached_analysis is not None and hotspot_age <= 2.0:
            result = {"ok": True, "analysis": cached_analysis}
            if include_frames:
                result["visibleFrame"] = visible
                result["thermalFrame"] = thermal
            return result

        adjusted_visible_quad = visible_quad
        if visible_quad is not None and manual_adjust:
            adjusted_visible_quad = adjust_quad(visible_quad, manual_adjust)

        if board_matrix is None and visible_quad is not None:
            board_matrix = board_matrix_from_quad(visible_quad)

        if visible_quad is not None and thermal_quad is not None:
            adjusted_thermal_quad = adjust_source_quad(thermal_quad, manual_adjust, thermal.shape)
            if matrix is not None:
                adjusted_matrix = compose_board_space_matrix(matrix, visible_quad, adjusted_visible_quad)
            elif rig_matrix is not None:
                adjusted_matrix = compose_board_space_matrix(rig_matrix, visible_quad, adjusted_visible_quad)
            else:
                adjusted_matrix = cv2.getPerspectiveTransform(adjusted_thermal_quad.astype(np.float32), adjusted_visible_quad.astype(np.float32))
        else:
            source = rig_matrix if rig_matrix is not None else matrix
            if source is None:
                return {"ok": False, "error": "hotspot mapping unavailable"}
            adjusted_matrix = np.asarray(source, dtype=np.float32)

        analysis = self._get_cached_hotspot_analysis(
            thermal,
            adjusted_matrix,
            board_matrix,
            visible_quad,
            thermal_quad,
            manual_adjust,
            yolo_detections,
            visible.shape,
        )
        if analysis is None:
            with self.calibration_lock:
                self.hotspot = None
                self.hotspot_analysis = None
            return {"ok": False, "error": "hotspot not found"}
        with self.calibration_lock:
            self.hotspot = analysis.get("imagePoint")
            self.hotspot_analysis = deepcopy(analysis)
        result = {"ok": True, "analysis": analysis}
        if include_frames:
            result["visibleFrame"] = visible
            result["thermalFrame"] = thermal
        return result

    def generate_diagnostic_report(self, llm_api_url=None, llm_mode="auto"):
        llm_mode = normalize_llm_mode(llm_mode)
        current_status = self.get_status()
        if (
            current_status.get("boardPose") != "manual-holes-pair"
            or not current_status.get("manualVisiblePose")
            or not current_status.get("manualThermalPose")
            or current_status.get("calibrationNotice")
        ):
            error_text = "manual fusion calibration is not valid; recalibrate visible and thermal before generating report"
            self._set_report_state(
                running=False,
                progress=0,
                stage="error",
                message="标定无效，请重新标定后再生成报告",
                finishedAt=time.time(),
                error=error_text,
            )
            return {"ok": False, "error": error_text}
        self._set_report_state(
            running=True,
            progress=3,
            stage="analysis",
            message="正在提取热区并定位器件",
            provider=None,
            providerLabel=None,
            startedAt=time.time(),
            finishedAt=None,
            error=None,
            pdfPath=None,
            pdfUrl=None,
        )
        self._set_report_state(requestedMode=llm_mode, requestedModeLabel=llm_mode_label(llm_mode))
        try:
            now = time.time()
            with self.calibration_lock:
                cached_analysis = deepcopy(self.hotspot_analysis)
                hotspot_age = now - float(self.hotspot_last_refresh or 0.0)
            visible_frame = self.visible.get_frame(copy_frame=True)
            thermal_frame = self.thermal.get_frame(copy_frame=True)

            if cached_analysis is not None and hotspot_age <= 3.0:
                analysis_result = {
                    "ok": True,
                    "analysis": cached_analysis,
                    "visibleFrame": visible_frame,
                    "thermalFrame": thermal_frame,
                }
            else:
                analysis_result = self.analyze_current_hotspot(include_frames=False)
                if analysis_result.get("ok"):
                    analysis_result["visibleFrame"] = visible_frame
                    analysis_result["thermalFrame"] = thermal_frame
            if not analysis_result.get("ok"):
                self._set_report_state(
                    running=False,
                    progress=0,
                    stage="error",
                    message="热区分析失败",
                    finishedAt=time.time(),
                    error=analysis_result.get("error"),
                )
                return analysis_result

            self._set_report_state(progress=16, stage="analysis", message="正在整理诊断上下文")
            report_payload = build_diagnostic_report_payload(analysis_result["analysis"], self.get_status())

            self._set_report_state(progress=28, stage="llm", message="正在选择诊断模型")
            if llm_api_url:
                try:
                    llm_summary = http_post_json(llm_api_url, report_payload, timeout=20)
                    if isinstance(llm_summary, dict):
                        llm_summary.setdefault("provider", "custom-api")
                        llm_summary.setdefault("transport", "api")
                        llm_summary.setdefault("providerLabel", "自定义 API")
                except Exception as exc:
                    llm_summary = {"ok": False, "error": str(exc), "provider": "custom-api", "transport": "api"}
            else:
                llm_summary = call_deepseek_summary(
                    report_payload,
                    progress_callback=self._set_report_state,
                    llm_mode=llm_mode,
                )
            if not isinstance(llm_summary, dict):
                llm_summary = build_fallback_llm_summary(report_payload)
            elif not llm_summary.get("ok"):
                requested_label = llm_mode_label(llm_mode)
                llm_summary = build_fallback_llm_summary(
                    report_payload,
                    base_result=llm_summary,
                    provider=llm_summary.get("provider") or "rules-fallback",
                    provider_label=f"{requested_label}不可用，已回退规则诊断",
                )
            llm_summary = sanitize_llm_temperature_fields(llm_summary, report_payload)
            report_payload["llm"] = llm_summary

            provider_label = llm_summary.get("providerLabel") if isinstance(llm_summary, dict) else None
            provider_name = llm_summary.get("provider") if isinstance(llm_summary, dict) else None
            self._set_report_state(
                progress=74,
                stage="images",
                message="正在整理报告图片",
                provider=provider_name,
                providerLabel=provider_label,
            )

            report_stem = f"diagnostic-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            visible_frame = analysis_result.get("visibleFrame")
            thermal_frame = analysis_result.get("thermalFrame")
            report_payload["reportImages"] = build_report_image_assets(
                report_payload.get("analysis"),
                visible_frame=visible_frame,
                thermal_frame=thermal_frame,
                stem=report_stem,
            )

            self._set_report_state(progress=88, stage="pdf", message="正在生成 PDF")
            pdf_path = render_diagnostic_report_pdf(report_payload)
            report_payload["pdfPath"] = str(pdf_path)
            report_payload["pdfName"] = pdf_path.name
            report_payload["pdfUrl"] = f"/reports/{pdf_path.name}"
            with self.calibration_lock:
                if self.hotspot_analysis is None:
                    self.hotspot_analysis = deepcopy(report_payload.get("analysis"))
            self._set_report_state(
                running=False,
                progress=100,
                stage="done",
                message="诊断报告已生成",
                finishedAt=time.time(),
                pdfPath=str(pdf_path),
                pdfUrl=report_payload["pdfUrl"],
                provider=provider_name,
                providerLabel=provider_label,
                error=None,
            )
            return {"ok": True, "report": report_payload}
        except Exception as exc:
            self._set_report_state(
                running=False,
                progress=0,
                stage="error",
                message="诊断报告生成失败",
                finishedAt=time.time(),
                error=str(exc),
            )
            return {"ok": False, "error": str(exc)}

    def request_recalibrate(self):
        with self.calibration_lock:
            self.display_annotations = True
            self.matrix = None
            self.visible_quad = None
            self.thermal_quad = None
            self.thermal_board_matrix = None
            self.thermal_rotation = "capture-cw"
            self.board_matrix = None
            self.board_pose_method = "none"
            self.hole_debug = {"method": "none", "accepted": False, "candidates": []}
            self.profile_match = {"profileId": None, "score": None, "matched": False}
            self.board_fingerprint = None
            self.calibration_notice = None
            self._clear_hotspot_runtime_cache()
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
            self.samples = 0
            self.recalibrate_requested = True
        return {"ok": True, "message": "recalibration requested"}

    def apply_runtime_board_model(self, model):
        payload = set_active_board_model(model, persist=True)
        clear_manual_pose_files()
        with self.calibration_lock:
            self.display_annotations = False
            self.manual_board_pose = None
            self.manual_thermal_pose = None
            self.matrix = None
            self.visible_quad = None
            self.thermal_quad = None
            self.thermal_board_matrix = None
            self.board_matrix = None
            self.board_pose_method = "none"
            self.hole_debug = {"method": "none", "accepted": False, "candidates": []}
            self.profile_match = {"profileId": payload.get("id"), "score": None, "matched": False}
            self.calibration_notice = None
            self._clear_hotspot_runtime_cache()
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
            self.recalibrate_requested = True
        return {"ok": True, "boardModel": payload}

    def clear_runtime_board_model(self):
        clear_runtime_board_model()
        payload = set_active_board_model(load_board_model_from_path(DEFAULT_BOARD_MODEL_PATH), persist=False)
        clear_manual_pose_files()
        with self.calibration_lock:
            self.display_annotations = False
            self.manual_board_pose = None
            self.manual_thermal_pose = None
            self.matrix = None
            self.visible_quad = None
            self.thermal_quad = None
            self.thermal_board_matrix = None
            self.board_matrix = None
            self.board_pose_method = "none"
            self.hole_debug = {"method": "none", "accepted": False, "candidates": []}
            self.profile_match = {"profileId": payload.get("id"), "score": None, "matched": False}
            self.calibration_notice = None
            self._clear_hotspot_runtime_cache()
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
            self.recalibrate_requested = True
        return {"ok": True, "boardModel": payload}

    def run_yolo_obb_once(self):
        with self.calibration_lock:
            if self.yolo_obb.get("running"):
                return {"ok": False, "error": "YOLO OBB detection already running"}
            self.yolo_obb["running"] = True
            self.yolo_obb["enabled"] = True
            self.yolo_obb["message"] = "running"
            save_yolo_obb_result(self.yolo_obb)

        frame = self.visible.get_frame()
        if frame is None:
            with self.calibration_lock:
                self.yolo_obb["running"] = False
                self.yolo_obb["available"] = False
                self.yolo_obb["message"] = "visible frame unavailable"
                save_yolo_obb_result(self.yolo_obb)
            return {"ok": False, "error": "visible frame unavailable"}

        try:
            with self.calibration_lock:
                visible_quad = None if self.visible_quad is None else np.asarray(self.visible_quad, dtype=np.float32).copy()
            board_crop, crop_origin = build_board_crop(frame, visible_quad, pad_ratio=0.05)
            image_path = ROOT / "yolo_obb_input.jpg"
            cv2.imwrite(str(image_path), board_crop)

            model_path = find_yolo_obb_model()
            if model_path is None:
                raise RuntimeError("no YOLO OBB model found on board")

            if model_path.suffix.lower() == ".rknn":
                detections = run_rknn_obb_detection(board_crop, model_path, conf_threshold=0.22, nms_threshold=0.10)
                for det in detections:
                    quad = np.asarray(det.get("quad", []), dtype=np.float32)
                    if quad.shape == (4, 2):
                        quad[:, 0] += crop_origin[0]
                        quad[:, 1] += crop_origin[1]
                        det["quad"] = quad.tolist()
                payload = {
                    "enabled": True,
                    "running": False,
                    "available": True,
                    "message": "ok",
                    "savedAt": time.time(),
                    "modelPath": str(model_path),
                    "count": len(detections),
                    "detections": detections,
                }
                save_yolo_obb_result(payload)
            else:
                result, model_path = run_yolo_obb_detection(image_path, YOLO_OBB_RESULT_PATH)
                if result.returncode != 0:
                    message = (result.stderr or result.stdout or "YOLO OBB failed").strip()
                    raise RuntimeError(message)
                payload = load_yolo_obb_result()
                for det in payload.get("detections", []):
                    quad = np.asarray(det.get("quad", []), dtype=np.float32)
                    if quad.shape == (4, 2):
                        quad[:, 0] += crop_origin[0]
                        quad[:, 1] += crop_origin[1]
                        det["quad"] = quad.tolist()
                payload["running"] = False
                payload["enabled"] = True
                payload["available"] = True
                payload["modelPath"] = str(model_path)
                payload["savedAt"] = time.time()
                save_yolo_obb_result(payload)
            with self.calibration_lock:
                self.yolo_obb = payload
                save_yolo_obb_result(self.yolo_obb)
                self._invalidate_overlay_cache()
            return {"ok": True, "result": deepcopy(payload)}
        except Exception as exc:
            with self.calibration_lock:
                self.yolo_obb["running"] = False
                self.yolo_obb["available"] = False
                self.yolo_obb["message"] = str(exc)
                self.yolo_obb["count"] = 0
                self.yolo_obb["detections"] = []
                save_yolo_obb_result(self.yolo_obb)
                self._invalidate_overlay_cache()
            return {"ok": False, "error": str(exc)}

    def set_manual_board_pose(self, image_points):
        image_points = order_quad(np.asarray(image_points, dtype=np.float32))
        if image_points.shape != (4, 2):
            return {"ok": False, "error": "expected four image points"}
        matrix = board_matrix_from_hole_points(image_points)
        if matrix is None:
            return {"ok": False, "error": "hole model unavailable"}
        quad = order_quad(board_points_to_image(matrix, board_corners_mm()).astype(np.float32))
        payload = save_manual_board_pose(image_points)
        with self.calibration_lock:
            self.manual_board_pose = {"imagePoints": image_points, "savedAt": payload["savedAt"]}
            self.visible_quad = quad
            self.board_matrix = matrix.astype(np.float32)
            self.board_pose_method = "manual-holes"
            self.calibration_notice = None
            self.hole_debug = {
                "method": "manual-holes",
                "accepted": True,
                "reason": "user-selected",
                "candidates": [
                    {"ref": ref, "best": {"point": [round(float(pt[0]), 2), round(float(pt[1]), 2)]}}
                    for ref, pt in zip(("hole3", "hole1", "hole2", "hole4"), image_points)
                ],
            }
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
        return {"ok": True, "pose": payload}

    def clear_manual_board_pose(self):
        try:
            if MANUAL_BOARD_POSE_PATH.exists():
                MANUAL_BOARD_POSE_PATH.unlink()
        except Exception:
            pass
        with self.calibration_lock:
            self.manual_board_pose = None
            self.display_annotations = False
            self.calibration_notice = None
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
            self.recalibrate_requested = True
        return {"ok": True, "message": "manual board pose cleared"}

    def set_manual_thermal_pose(self, image_points):
        image_points = order_quad(np.asarray(image_points, dtype=np.float32))
        if image_points.shape != (4, 2):
            return {"ok": False, "error": "expected four image points"}
        payload = save_manual_thermal_pose(image_points)
        with self.calibration_lock:
            self.manual_thermal_pose = {"imagePoints": image_points, "savedAt": payload["savedAt"]}
            self.calibration_notice = None
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
        return {"ok": True, "pose": payload}

    def clear_manual_thermal_pose(self):
        try:
            if MANUAL_THERMAL_POSE_PATH.exists():
                MANUAL_THERMAL_POSE_PATH.unlink()
        except Exception:
            pass
        with self.calibration_lock:
            self.manual_thermal_pose = None
            self.display_annotations = False
            self.calibration_notice = None
            self._invalidate_debug_frame_cache()
            self._invalidate_overlay_cache()
            self.recalibrate_requested = True
        return {"ok": True, "message": "manual thermal pose cleared"}

    def _run(self):
        frames = 0
        started = time.time()
        while self.running:
            visible = self.visible.get_frame(copy_frame=False)
            thermal = self.thermal.get_frame(copy_frame=False)
            if visible is None and thermal is None:
                time.sleep(0.03)
                continue

            self._maybe_calibrate(visible, thermal)
            render_visible, render_scale_x, render_scale_y = self._render_preview_context(visible)
            output_frame, status = self._build_frame(visible, thermal, render_visible, render_scale_x, render_scale_y)
            ok, encoded = cv2.imencode(
                ".jpg",
                output_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue

            frames += 1
            status["fps"] = frames / max(time.time() - started, 1e-6)
            status["online"] = True

            with self.lock:
                self.latest_jpeg = encoded.tobytes()
                self.latest_status = status
                self._maybe_refresh_debug_jpeg(visible)

            time.sleep(self.frame_interval)

    def _prepare_output_frame(self, frame):
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if width <= self.preview_max_width:
            return frame
        scale = self.preview_max_width / max(float(width), 1.0)
        target_size = (
            self.preview_max_width,
            max(1, int(round(height * scale))),
        )
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _render_preview_context(self, visible):
        if visible is None:
            return None, 1.0, 1.0
        height, width = visible.shape[:2]
        if width <= self.preview_max_width:
            return visible, 1.0, 1.0
        scale = self.preview_max_width / max(float(width), 1.0)
        preview = cv2.resize(
            visible,
            (self.preview_max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return preview, scale, scale

    def _scale_quad(self, quad, scale_x, scale_y):
        if quad is None:
            return None
        scaled = np.asarray(quad, dtype=np.float32).copy()
        scaled[:, 0] *= float(scale_x)
        scaled[:, 1] *= float(scale_y)
        return scaled

    def _scale_homography_target(self, matrix, scale_x, scale_y):
        if matrix is None:
            return None
        scale_matrix = np.array(
            [[float(scale_x), 0.0, 0.0], [0.0, float(scale_y), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return scale_matrix @ np.asarray(matrix, dtype=np.float32)

    def _build_frame(self, visible, thermal, render_visible=None, render_scale_x=1.0, render_scale_y=1.0):
        status = {
            "mode": "waiting",
            "calibrated": False,
            "visibleBorder": False,
            "thermalBorder": False,
            "thermalRotation": self.thermal_rotation,
            "samples": self.samples,
            "alpha": float(self.alpha),
            "visibleQuad": None,
            "thermalQuad": None,
            "manualAdjust": dict(self.manual_adjust),
            "profileMatch": dict(self.profile_match),
            "hotspot": None,
            "hotspotAnalysis": deepcopy(self.hotspot_analysis),
            "boardPose": self.board_pose_method,
            "boardModelId": BOARD_MODEL.get("id"),
            "boardSourceFile": BOARD_MODEL.get("sourceFile"),
            "componentCount": len(BOARD_MODEL.get("components", [])),
            "holeDebug": dict(self.hole_debug),
            "yoloObb": summarize_yolo_obb_state(self.yolo_obb),
        }

        if visible is None:
            base = np.zeros((720, 1280, 3), dtype=np.uint8)
            if thermal is not None:
                rect = fit_rect(thermal.shape[1], thermal.shape[0], base.shape[1], base.shape[0])
                paste_resized(base, thermal, rect)
            status["mode"] = "thermal-only"
            return base, status

        if render_visible is None:
            render_visible = visible
        base = render_visible.copy()
        if thermal is None:
            status["mode"] = "visible-only"
            draw_label(base, "thermal stream offline", (24, 46), (0, 200, 255))
            return base, status

        matrix = self.matrix
        rig_matrix = self.rig_matrix
        visible_quad = self.visible_quad
        thermal_quad = self.thermal_quad
        board_matrix = self.board_matrix
        thermal_board_matrix = self.thermal_board_matrix
        thermal_rotation = self.thermal_rotation
        manual_adjust = dict(self.manual_adjust)
        show_annotations = bool(self.display_annotations)
        status["calibrated"] = matrix is not None or rig_matrix is not None
        status["visibleBorder"] = visible_quad is not None
        status["thermalBorder"] = thermal_quad is not None
        status["thermalRotation"] = thermal_rotation
        status["samples"] = self.samples
        status["alpha"] = float(self.alpha)
        status["manualAdjust"] = manual_adjust
        status["profileMatch"] = dict(self.profile_match)
        status["hotspotAnalysis"] = deepcopy(self.hotspot_analysis)
        status["boardPose"] = self.board_pose_method
        status["manualVisiblePose"] = self.manual_board_pose is not None
        status["manualThermalPose"] = self.manual_thermal_pose is not None
        status["calibrationNotice"] = self.calibration_notice
        if visible_quad is not None:
            status["visibleQuad"] = np.round(visible_quad, 2).tolist()
        if thermal_quad is not None:
            status["thermalQuad"] = np.round(thermal_quad, 2).tolist()

        manual_visible_points = None if self.manual_board_pose is None else order_quad(np.asarray(self.manual_board_pose["imagePoints"], dtype=np.float32))
        manual_thermal_points = None if self.manual_thermal_pose is None else order_quad(np.asarray(self.manual_thermal_pose["imagePoints"], dtype=np.float32))

        render_visible_quad = self._scale_quad(visible_quad, render_scale_x, render_scale_y)

        if show_annotations and render_visible_quad is not None:
            cv2.polylines(base, [render_visible_quad.astype(np.int32)], True, (0, 255, 120), 3)

        if visible_quad is not None and (matrix is not None or rig_matrix is not None):
            adjusted_visible_quad = adjust_quad(visible_quad, manual_adjust)
            render_adjusted_visible_quad = self._scale_quad(adjusted_visible_quad, render_scale_x, render_scale_y)
            source_matrix = matrix if matrix is not None else rig_matrix
            if manual_visible_points is not None and manual_thermal_points is not None:
                source_quad = thermal_quad if thermal_quad is not None else manual_thermal_points
                adjusted_thermal_quad = adjust_source_quad(source_quad, manual_adjust, thermal.shape)
                if source_matrix is None and (board_matrix is None or thermal_board_matrix is None):
                    status["mode"] = "visible-only"
                    draw_label(base, "fusion recalibrating", (24, 46), (0, 200, 255))
                    return base, status
                if board_matrix is not None and thermal_board_matrix is not None:
                    adjusted_matrix = compose_board_matrix(
                        board_matrix,
                        thermal_board_matrix,
                        visible_quad,
                        adjusted_visible_quad,
                    )
                else:
                    adjusted_matrix = compose_board_space_matrix(source_matrix, visible_quad, adjusted_visible_quad)
                render_matrix = self._scale_homography_target(adjusted_matrix, render_scale_x, render_scale_y)
                mask = np.zeros(base.shape[:2], dtype=np.uint8)
                if render_adjusted_visible_quad is not None:
                    cv2.fillConvexPoly(mask, np.round(render_adjusted_visible_quad).astype(np.int32), 255)
                else:
                    source_mask = np.zeros(thermal.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(source_mask, np.round(adjusted_thermal_quad).astype(np.int32), 255)
                    mask = cv2.warpPerspective(
                        source_mask,
                        render_matrix,
                        (base.shape[1], base.shape[0]),
                    )
                status["boardPose"] = "manual-holes-pair"
            elif matrix is not None and thermal_quad is not None:
                adjusted_thermal_quad = adjust_source_quad(thermal_quad, manual_adjust, thermal.shape)
                adjusted_matrix = cv2.getPerspectiveTransform(adjusted_thermal_quad.astype(np.float32), adjusted_visible_quad.astype(np.float32))
                source_mask = np.zeros(thermal.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(source_mask, np.round(adjusted_thermal_quad).astype(np.int32), 255)
                render_matrix = self._scale_homography_target(adjusted_matrix, render_scale_x, render_scale_y)
                mask = cv2.warpPerspective(
                    source_mask,
                    render_matrix,
                    (base.shape[1], base.shape[0]),
                )
            else:
                adjusted_matrix = compose_board_space_matrix(rig_matrix, visible_quad, adjusted_visible_quad)
                source_mask = np.full(thermal.shape[:2], 255, dtype=np.uint8)
                render_matrix = self._scale_homography_target(adjusted_matrix, render_scale_x, render_scale_y)
                mask = cv2.warpPerspective(
                    source_mask,
                    render_matrix,
                    (base.shape[1], base.shape[0]),
                )
            warped = cv2.warpPerspective(thermal, render_matrix, (base.shape[1], base.shape[0]))
            base = alpha_blend_mask(base, warped, mask, self.alpha)
            if show_annotations:
                cv2.polylines(base, [render_visible_quad.astype(np.int32)], True, (0, 255, 120), 3)
                cv2.polylines(base, [render_adjusted_visible_quad.astype(np.int32)], True, (0, 210, 255), 2)
            hotspot_analysis = deepcopy(self.hotspot_analysis)
            if hotspot_analysis is not None and hotspot_analysis.get("imagePoint") is not None:
                mapped = hotspot_analysis["imagePoint"]
                status["hotspot"] = [round(float(mapped[0]), 2), round(float(mapped[1]), 2)]
                status["hotspotAnalysis"] = deepcopy(hotspot_analysis)
            status["mode"] = "visual-locked-overlay" if matrix is not None else "rig-locked-overlay"
        else:
            rect = fit_rect(thermal.shape[1], thermal.shape[0], base.shape[1], base.shape[0])
            overlay = np.zeros_like(base)
            mask = np.zeros(base.shape[:2], dtype=np.uint8)
            paste_resized(overlay, thermal, rect)
            mask[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = 255
            base = alpha_blend_mask(base, overlay, mask, self.alpha)
            status["mode"] = "fallback-fit"

        if board_matrix is None and visible_quad is not None:
            board_matrix = board_matrix_from_quad(visible_quad)
        if show_annotations and board_matrix is not None:
            board_overlay = self._get_board_overlay(base.shape, self._scale_homography_target(board_matrix, render_scale_x, render_scale_y))
            if board_overlay is not None:
                apply_cached_overlay(base, board_overlay)

        if show_annotations and self.yolo_obb.get("enabled") and self.yolo_obb.get("detections"):
            obb_overlay = self._get_obb_overlay(base.shape, render_scale_x, render_scale_y)
            if obb_overlay is not None:
                apply_cached_overlay(base, obb_overlay)

        if show_annotations:
            draw_label(
                base,
                f"fusion: {status['mode']} | board: {status['boardPose']} | components: {status['componentCount']} | samples: {status['samples']}",
                (24, 46),
                (255, 255, 255),
            )
        return base, status

    def _maybe_calibrate(self, visible, thermal):
        if visible is None or thermal is None:
            return
        self._invalidate_stale_manual_pose_if_needed(visible)
        with self.calibration_lock:
            needs_calibration = self.recalibrate_requested or self.matrix is None
        if not needs_calibration:
            return

        manual_visible_points = None if self.manual_board_pose is None else order_quad(np.asarray(self.manual_board_pose["imagePoints"], dtype=np.float32))
        manual_thermal_points = None if self.manual_thermal_pose is None else order_quad(np.asarray(self.manual_thermal_pose["imagePoints"], dtype=np.float32))

        if manual_visible_points is not None and manual_thermal_points is not None:
            board_matrix = board_matrix_from_hole_points(manual_visible_points)
            thermal_board_matrix = board_matrix_from_hole_points(manual_thermal_points)
            if board_matrix is not None and thermal_board_matrix is not None:
                visible_quad = order_quad(board_points_to_image(board_matrix, board_corners_mm()).astype(np.float32))
                thermal_quad = order_quad(board_points_to_image(thermal_board_matrix, board_corners_mm()).astype(np.float32))
                board_matrix = board_matrix_from_quad(visible_quad).astype(np.float32)
                thermal_board_matrix = board_matrix_from_quad(thermal_quad).astype(np.float32)
                matrix = cv2.getPerspectiveTransform(
                    manual_thermal_points.astype(np.float32),
                    manual_visible_points.astype(np.float32),
                )
                fingerprint = fingerprint_from_quads(visible_quad, manual_thermal_points, visible.shape[:2], thermal.shape[:2])
                with self.calibration_lock:
                    self.visible_quad = visible_quad
                    self.thermal_quad = thermal_quad
                    self.board_matrix = board_matrix
                    self.thermal_board_matrix = thermal_board_matrix
                    self.board_pose_method = "manual-holes-pair"
                    self.hole_debug = {
                        "method": "manual-holes-pair",
                        "accepted": True,
                        "reason": "saved-user-selected",
                        "candidates": [
                            {"ref": ref, "best": {"point": [round(float(pt[0]), 2), round(float(pt[1]), 2)]}}
                            for ref, pt in zip(("hole3", "hole1", "hole2", "hole4"), manual_visible_points)
                        ],
                    }
                    self.thermal_rotation = "capture-cw"
                    self.board_fingerprint = fingerprint
                    self.profile_match = {"profileId": BOARD_MODEL["id"], "score": None, "matched": True}
                    self.matrix = matrix.astype(np.float32)
                    self.rig_matrix = matrix.astype(np.float32)
                    self.samples = 4
                    self.recalibrate_requested = False
                save_rig_calibration(matrix.astype(np.float32), visible_quad, manual_thermal_points.astype(np.float32))
                return

        paired_samples = []
        deadline = time.time() + 2.0
        while self.running and time.time() < deadline and len(paired_samples) < 12:
            visible_frame = self.visible.get_frame(copy_frame=False)
            thermal_frame = self.thermal.get_frame(copy_frame=False)
            if visible_frame is None or thermal_frame is None:
                time.sleep(0.03)
                continue

            v_quad = find_board_quad(visible_frame, prefer_color=True)
            if v_quad is not None:
                t_quad = find_board_quad(thermal_frame, prefer_color=False)
                if t_quad is not None:
                    paired_samples.append((v_quad, t_quad))
            time.sleep(0.04)

        if len(paired_samples) < 4:
            with self.calibration_lock:
                self.recalibrate_requested = False
                self.samples = len(paired_samples)
            return

        paired_visible = [sample[0] for sample in paired_samples]
        paired_thermal = [sample[1] for sample in paired_samples]
        v_locked = np.median(np.stack(paired_visible), axis=0).astype(np.float32)
        t_locked = np.median(np.stack(paired_thermal), axis=0).astype(np.float32)
        v_locked, board_matrix, pose_method, hole_debug = estimate_board_pose_from_model(visible, v_locked)
        if self.manual_board_pose is not None:
            manual_points = order_quad(np.asarray(self.manual_board_pose["imagePoints"], dtype=np.float32))
            manual_matrix = board_matrix_from_hole_points(manual_points)
            if manual_matrix is not None:
                board_matrix = manual_matrix.astype(np.float32)
                v_locked = order_quad(board_points_to_image(board_matrix, board_corners_mm()).astype(np.float32))
                pose_method = "manual-holes"
                hole_debug = {
                    "method": "manual-holes",
                    "accepted": True,
                    "reason": "saved-user-selected",
                    "candidates": [
                        {"ref": ref, "best": {"point": [round(float(pt[0]), 2), round(float(pt[1]), 2)]}}
                        for ref, pt in zip(("hole3", "hole1", "hole2", "hole4"), manual_points)
                    ],
                }
        if self.manual_thermal_pose is not None:
            t_locked = order_quad(np.asarray(self.manual_thermal_pose["imagePoints"], dtype=np.float32))
        fingerprint = fingerprint_from_quads(v_locked, t_locked, visible.shape[:2], thermal.shape[:2])
        matched_adjust, profile_match = default_manual_adjust(), {
            "profileId": BOARD_MODEL["id"],
            "score": None,
            "matched": True,
        }
        matrix = cv2.getPerspectiveTransform(t_locked, v_locked)
        save_rig_calibration(matrix, v_locked, t_locked)

        with self.calibration_lock:
            self.visible_quad = v_locked
            self.thermal_quad = t_locked
            self.board_matrix = board_matrix
            self.board_pose_method = pose_method
            self.hole_debug = hole_debug
            self.thermal_rotation = "capture-cw"
            self.board_fingerprint = fingerprint
            self.profile_match = profile_match
            self.manual_adjust = matched_adjust
            self.matrix = matrix
            self.rig_matrix = matrix
            self.samples = len(paired_samples)
            self.recalibrate_requested = False


def draw_label(frame, text, origin, color):
    x, y = origin
    cv2.rectangle(frame, (x - 8, y - 28), (x + 760, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def board_corners_mm():
    width = float(BOARD_MODEL["widthMm"])
    height = float(BOARD_MODEL["heightMm"])
    # CAD/PickAndPlace coordinates use a bottom-left origin with Y increasing
    # upward. Image quads are ordered top-left, top-right, bottom-right,
    # bottom-left, so the board-space Y axis must be flipped here.
    return np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=np.float32)


def board_matrix_from_quad(quad):
    return cv2.getPerspectiveTransform(board_corners_mm(), np.asarray(quad, dtype=np.float32))


def board_matrix_from_hole_points(image_points):
    holes = BOARD_MODEL.get("holeCentersMm", {})
    refs = ("hole3", "hole1", "hole2", "hole4")
    if all(ref in holes for ref in refs):
        model_points = np.asarray(
            [[float(holes[ref]["x"]), float(holes[ref]["y"])] for ref in refs],
            dtype=np.float32,
        )
    else:
        fallback_holes = {}
        try:
            fallback_holes = load_board_model_from_path(DEFAULT_BOARD_MODEL_PATH).get("holeCentersMm", {})
        except Exception:
            fallback_holes = {}
        if all(ref in fallback_holes for ref in refs):
            model_points = np.asarray(
                [[float(fallback_holes[ref]["x"]), float(fallback_holes[ref]["y"])] for ref in refs],
                dtype=np.float32,
            )
        else:
            # Last-resort fallback for uploaded board profiles without hole metadata:
            # treat the selected four points as board corners so calibration does not
            # collapse into an expensive recalibration loop.
            model_points = board_corners_mm()
    return cv2.getPerspectiveTransform(model_points, np.asarray(image_points, dtype=np.float32))


def board_points_to_image(matrix, points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)


def image_points_to_board(matrix, points):
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float32))
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, inverse).reshape(-1, 2)


def score_hole_board_support(frame, center, radius_px):
    x, y = int(round(float(center[0]))), int(round(float(center[1])))
    radius_px = float(np.clip(radius_px, 14.0, 55.0))
    pad = int(round(radius_px * 1.9))
    height, width = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + pad)
    y2 = min(height, y + pad)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return -1.0

    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    cx = x - x1
    cy = y - y1
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus = (dist >= radius_px * 0.75) & (dist <= radius_px * 1.85)
    if not np.any(annulus):
        return -1.0

    green = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 105)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 2] >= 20)
    )
    dark_board_detail = (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] <= 150)
    red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 80)
    paper = (hsv[:, :, 1] <= 35) & (hsv[:, :, 2] >= 135)

    support = float(np.mean((green | dark_board_detail)[annulus]))
    red_penalty = float(np.mean(red[annulus]))
    paper_penalty = float(np.mean(paper[annulus]))
    return support - red_penalty * 1.5 - paper_penalty * 0.8


def point_inside_quad(point, quad, margin_px=0.0):
    pts = np.asarray(quad, dtype=np.float32)
    if margin_px:
        center = pts.mean(axis=0)
        expanded = center + (pts - center) * (1.0 + float(margin_px) / max(np.linalg.norm(pts[1] - pts[0]), 1.0))
        pts = expanded.astype(np.float32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def hole_ring_score(frame, center, radius_px):
    x, y = int(round(float(center[0]))), int(round(float(center[1])))
    radius_px = float(np.clip(radius_px, 10.0, 45.0))
    pad = int(round(radius_px * 2.2))
    height, width = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + pad)
    y2 = min(height, y + pad)
    if x2 <= x1 + 12 or y2 <= y1 + 12:
        return -10.0

    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    cx = x - x1
    cy = y - y1
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = dist <= radius_px * 0.55
    ring = (dist >= radius_px * 0.75) & (dist <= radius_px * 1.65)
    outer = (dist >= radius_px * 1.65) & (dist <= radius_px * 2.15)
    if not np.any(inner) or not np.any(ring):
        return -10.0

    green = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 105)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 2] >= 20)
    )
    light_ring = (hsv[:, :, 1] <= 75) & (hsv[:, :, 2] >= 105)
    dark_inner = gray <= 115
    paper = (hsv[:, :, 1] <= 35) & (hsv[:, :, 2] >= 145)
    red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 80)

    inner_dark = float(np.mean(dark_inner[inner]))
    ring_light = float(np.mean(light_ring[ring]))
    ring_green = float(np.mean(green[ring]))
    outer_board = float(np.mean(green[outer])) if np.any(outer) else 0.0
    paper_penalty = float(np.mean(paper[ring | outer])) if np.any(ring | outer) else 0.0
    red_penalty = float(np.mean(red[ring | outer])) if np.any(ring | outer) else 0.0
    contrast = (float(np.mean(gray[ring])) - float(np.mean(gray[inner]))) / 80.0
    return inner_dark * 2.0 + ring_light * 1.1 + ring_green * 1.2 + outer_board * 1.0 + contrast - paper_penalty * 2.2 - red_penalty * 1.8


def collect_hole_candidates(frame, predicted, radius_px):
    x, y = float(predicted[0]), float(predicted[1])
    radius_px = float(np.clip(radius_px, 18.0, 70.0))
    pad = int(round(radius_px * 4.4))
    height, width = frame.shape[:2]
    x1 = max(0, int(round(x)) - pad)
    y1 = max(0, int(round(y)) - pad)
    x2 = min(width, int(round(x)) + pad)
    y2 = min(height, int(round(y)) + pad)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    candidates = []

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(radius_px * 1.5)),
        param1=90,
        param2=12,
        minRadius=max(6, int(radius_px * 0.45)),
        maxRadius=max(10, int(radius_px * 1.9)),
    )
    if circles is not None:
        for cx, cy, detected_radius in circles[0]:
            ix = x1 + float(cx)
            iy = y1 + float(cy)
            dist = np.linalg.norm(np.array([ix - x, iy - y], dtype=np.float32))
            if dist > radius_px * 4.4:
                continue
            radius_penalty = abs(float(detected_radius) - radius_px) / max(radius_px, 1.0)
            support = score_hole_board_support(frame, (ix, iy), detected_radius)
            if support < -0.12:
                continue
            score = 5.0 + support * 4.0 - dist / max(radius_px * 3.6, 1.0) - radius_penalty * 0.7
            candidates.append((score, np.array([ix, iy], dtype=np.float32)))

    for mode in ("dark", "bright"):
        threshold_type = cv2.THRESH_BINARY_INV if mode == "dark" else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blur, 0, 255, threshold_type + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < radius_px * radius_px * 0.10 or area > radius_px * radius_px * 5.5:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < radius_px * 0.45 or bh < radius_px * 0.45 or bw > radius_px * 2.8 or bh > radius_px * 2.8:
                continue
            cx = x1 + bx + bw / 2.0
            cy = y1 + by + bh / 2.0
            dist = np.linalg.norm(np.array([cx - x, cy - y], dtype=np.float32))
            if dist > radius_px * 4.4:
                continue
            support = score_hole_board_support(frame, (cx, cy), max(bw, bh) * 0.5)
            if support < -0.12:
                continue
            score = circularity * 4.0 + support * 4.0 + area / max(radius_px * radius_px, 1.0) - dist / max(radius_px * 2.2, 1.0)
            candidates.append((score, np.array([cx, cy], dtype=np.float32)))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    deduped = []
    for score, point in candidates:
        if any(np.linalg.norm(point - kept_point) < radius_px * 0.8 for _, kept_point in deduped):
            continue
        deduped.append((score, point))
        if len(deduped) >= 6:
            break
    return deduped


def collect_strict_hole_candidates(frame, predicted, radius_px, board_quad):
    x, y = float(predicted[0]), float(predicted[1])
    radius_px = float(np.clip(radius_px, 10.0, 42.0))
    pad = int(round(radius_px * 3.2))
    height, width = frame.shape[:2]
    x1 = max(0, int(round(x)) - pad)
    y1 = max(0, int(round(y)) - pad)
    x2 = min(width, int(round(x)) + pad)
    y2 = min(height, int(round(y)) + pad)
    if x2 <= x1 + 16 or y2 <= y1 + 16:
        return []

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates = []

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(14, int(radius_px * 1.3)),
        param1=85,
        param2=14,
        minRadius=max(5, int(radius_px * 0.45)),
        maxRadius=max(8, int(radius_px * 1.35)),
    )
    if circles is not None:
        for cx, cy, detected_radius in circles[0]:
            point = np.array([x1 + float(cx), y1 + float(cy)], dtype=np.float32)
            if not point_inside_quad(point, board_quad, margin_px=8.0):
                continue
            dist = float(np.linalg.norm(point - np.array([x, y], dtype=np.float32)))
            if dist > radius_px * 3.0:
                continue
            ring = hole_ring_score(frame, point, detected_radius)
            support = score_hole_board_support(frame, point, detected_radius)
            radius_penalty = abs(float(detected_radius) - radius_px) / max(radius_px, 1.0)
            score = ring + support * 1.4 - dist / max(radius_px, 1.0) * 1.0 - radius_penalty * 1.4
            if score >= 0.90:
                candidates.append((score, point, float(detected_radius), "circle"))

    for threshold_type, mode in ((cv2.THRESH_BINARY_INV, "dark"), (cv2.THRESH_BINARY, "bright")):
        _, mask = cv2.threshold(blur, 0, 255, threshold_type + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < radius_px * radius_px * 0.18 or area > radius_px * radius_px * 3.6:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.38:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < radius_px * 0.55 or bh < radius_px * 0.55 or bw > radius_px * 2.35 or bh > radius_px * 2.35:
                continue
            point = np.array([x1 + bx + bw / 2.0, y1 + by + bh / 2.0], dtype=np.float32)
            if not point_inside_quad(point, board_quad, margin_px=8.0):
                continue
            dist = float(np.linalg.norm(point - np.array([x, y], dtype=np.float32)))
            if dist > radius_px * 3.0:
                continue
            est_radius = max(bw, bh) * 0.5
            ring = hole_ring_score(frame, point, est_radius)
            support = score_hole_board_support(frame, point, est_radius)
            score = ring + circularity * 1.4 + support - dist / max(radius_px, 1.0) * 1.0
            if score >= 0.90:
                candidates.append((score, point, float(est_radius), mode))

    candidates.sort(key=lambda item: item[0], reverse=True)
    deduped = []
    for score, point, detected_radius, mode in candidates:
        if any(np.linalg.norm(point - kept[1]) < radius_px * 0.55 for kept in deduped):
            continue
        deduped.append((score, point, detected_radius, mode))
        if len(deduped) >= 4:
            break
    return deduped


def refine_hole_center(frame, predicted, radius_px):
    candidates = collect_hole_candidates(frame, predicted, radius_px)
    if not candidates:
        return None
    best_score, best_point = candidates[0]
    if best_score < 1.7:
        return None
    return best_point


def estimate_board_pose_from_model(frame, coarse_quad):
    coarse_quad = np.asarray(coarse_quad, dtype=np.float32)
    coarse_quad = refine_visible_quad_by_pcb_edges(frame, coarse_quad)
    coarse_matrix = board_matrix_from_quad(coarse_quad)
    holes = BOARD_MODEL.get("holeCentersMm", {})
    debug = {
        "method": "outline",
        "accepted": False,
        "reason": "not-run",
        "candidates": [],
        "score": None,
        "errorPx": None,
    }
    if len(holes) < 4:
        debug["reason"] = "hole-model-missing"
        return coarse_quad, coarse_matrix, "quad", debug

    width_px = max(
        np.linalg.norm(coarse_quad[1] - coarse_quad[0]),
        np.linalg.norm(coarse_quad[2] - coarse_quad[3]),
    )
    height_px = max(
        np.linalg.norm(coarse_quad[3] - coarse_quad[0]),
        np.linalg.norm(coarse_quad[2] - coarse_quad[1]),
    )
    radius_px = width_px * 2.25 / max(float(BOARD_MODEL["widthMm"]), 1.0)

    hole_order = ("hole3", "hole1", "hole2", "hole4")
    candidate_sets = []
    model_points_by_ref = []
    best_by_ref = {}
    for ref in ("hole3", "hole1", "hole2", "hole4"):
        hole = holes.get(ref)
        if not hole:
            continue
        model_point = np.array([float(hole["x"]), float(hole["y"])], dtype=np.float32)
        predicted = board_points_to_image(coarse_matrix, [model_point])[0]
        candidates = collect_strict_hole_candidates(frame, predicted, radius_px, coarse_quad)
        filtered = []
        for score, point, detected_radius, mode in candidates:
            dx = abs(float(point[0] - predicted[0]))
            dy = abs(float(point[1] - predicted[1]))
            if dx > width_px * 0.095 or dy > height_px * 0.145:
                continue
            filtered.append((score, point, detected_radius, mode))
        candidates = filtered
        debug["candidates"].append(
            {
                "ref": ref,
                "predicted": [round(float(predicted[0]), 2), round(float(predicted[1]), 2)],
                "count": len(candidates),
                "best": None
                if not candidates
                else {
                    "score": round(float(candidates[0][0]), 3),
                    "point": [round(float(candidates[0][1][0]), 2), round(float(candidates[0][1][1]), 2)],
                    "radius": round(float(candidates[0][2]), 2),
                    "mode": candidates[0][3],
                },
            }
        )
        if not candidates:
            debug["reason"] = f"missing-{ref}"
        else:
            best_by_ref[ref] = (model_point, candidates[0][1], candidates[0][0])
            candidate_sets.append(candidates[:5])
            model_points_by_ref.append(model_point)
            continue
        candidate_sets.append([])
        model_points_by_ref.append(model_point)

    if len([items for items in candidate_sets if items]) == 4:
        best = None
        model_points = np.asarray(model_points_by_ref, dtype=np.float32)
        for combo in product(*candidate_sets):
            candidate_score = sum(item[0] for item in combo)
            image_points = np.asarray([item[1] for item in combo], dtype=np.float32)
            tl, tr, br, bl = image_points
            top = np.linalg.norm(tr - tl)
            bottom = np.linalg.norm(br - bl)
            left = np.linalg.norm(bl - tl)
            right = np.linalg.norm(br - tr)
            if min(top, bottom, left, right) <= 1e-6:
                continue
            aspect = ((top + bottom) * 0.5) / max((left + right) * 0.5, 1e-6)
            target_aspect = 72.563 / 42.745
            aspect_penalty = abs(np.log(aspect / target_aspect)) * 8.0
            parallel_penalty = (
                abs(top - bottom) / max(top, bottom)
                + abs(left - right) / max(left, right)
            ) * 3.0
            ordering_penalty = 0.0
            if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
                ordering_penalty += 30.0
            matrix = cv2.getPerspectiveTransform(model_points, image_points)
            quad = order_quad(board_points_to_image(matrix, board_corners_mm()).astype(np.float32))
            quad_center_error = float(np.linalg.norm(quad_center(quad) - quad_center(coarse_quad)))
            quad_size_error = abs(quad_aspect(quad) - BOARD_MODEL["aspect"]) / max(BOARD_MODEL["aspect"], 1e-6)
            corner_error = float(np.mean(np.linalg.norm(quad - coarse_quad, axis=1)))
            if quad_center_error > width_px * 0.090:
                continue
            if corner_error > width_px * 0.090:
                continue
            if quad_size_error > 0.10:
                continue
            if not all(point_inside_quad(point, quad, margin_px=4.0) for point in image_points):
                continue
            score = (
                candidate_score
                - aspect_penalty
                - parallel_penalty
                - ordering_penalty
                - quad_center_error * 0.04
                - corner_error * 0.03
            )
            if best is None or score > best[0]:
                best = (score, matrix, image_points, quad, corner_error)

        if best is not None and best[0] > 6.0 and best[4] <= width_px * 0.085:
            matrix = best[1]
            quad = best[3].astype(np.float32)
            debug["method"] = "holes-4"
            debug["accepted"] = True
            debug["reason"] = "ok"
            debug["score"] = round(float(best[0]), 3)
            debug["errorPx"] = round(float(best[4]), 2)
            return quad, matrix.astype(np.float32), "holes-4", debug

    if len(best_by_ref) >= 3:
        best_three = None
        for refs in (("hole3", "hole1", "hole2"), ("hole3", "hole2", "hole4"), ("hole1", "hole2", "hole4"), ("hole3", "hole1", "hole4")):
            if not all(ref in best_by_ref for ref in refs):
                continue
            src = np.asarray([best_by_ref[ref][0] for ref in refs], dtype=np.float32)
            dst = np.asarray([best_by_ref[ref][1] for ref in refs], dtype=np.float32)
            affine = cv2.getAffineTransform(src, dst)
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
            quad = order_quad(board_points_to_image(matrix, board_corners_mm()).astype(np.float32))
            if quad_aspect(quad) < BOARD_MODEL["aspect"] * 0.88 or quad_aspect(quad) > BOARD_MODEL["aspect"] * 1.12:
                continue
            corner_error = float(np.mean(np.linalg.norm(quad - coarse_quad, axis=1)))
            if corner_error > width_px * 0.11:
                continue
            detected = np.asarray([best_by_ref[ref][1] for ref in refs], dtype=np.float32)
            projected = board_points_to_image(matrix, [best_by_ref[ref][0] for ref in refs])
            reproj = float(np.mean(np.linalg.norm(projected - detected, axis=1)))
            if reproj > max(2.5, radius_px * 0.18):
                continue
            score = sum(float(best_by_ref[ref][2]) for ref in refs) - corner_error * 0.03 - reproj * 1.5
            if best_three is None or score > best_three[0]:
                best_three = (score, matrix, quad, refs, corner_error, reproj)
        if best_three is not None and best_three[0] > 4.0:
            debug["method"] = "holes-3"
            debug["accepted"] = True
            debug["reason"] = "one-hole-occluded"
            debug["score"] = round(float(best_three[0]), 3)
            debug["errorPx"] = round(float(best_three[4]), 2)
            debug["usedRefs"] = list(best_three[3])
            return best_three[2].astype(np.float32), best_three[1].astype(np.float32), "holes-3", debug

    debug["reason"] = "geometry-rejected"
    return coarse_quad, coarse_matrix, "outline", debug


def draw_text_with_backing(frame, text, origin, color, scale=0.48, thickness=1):
    x, y = int(round(origin[0])), int(round(origin[1]))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x - 2, y - th - 3), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def component_net_info(component):
    nets = BOARD_MODEL.get("nets", {})
    if not isinstance(nets, dict):
        return []
    ref = str(component.get("ref", "")).strip()
    if not ref:
        return []
    matches = []
    for net_name, payload in nets.items():
        nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        for node in nodes:
            if str(node.get("ref", "")).strip() == ref:
                matches.append(
                    {
                        "net": net_name,
                        "pin": node.get("pin"),
                        "name": node.get("name"),
                    }
                )
    return matches


def build_component_context(component, neighbor_limit=6, net_limit=4):
    if not isinstance(component, dict):
        return {}
    ref = str(component.get("ref", "")).strip()
    if not ref:
        return {}

    source_component = None
    for item in BOARD_MODEL.get("components", []):
        if str(item.get("ref", "")).strip() == ref:
            source_component = dict(item)
            break
    merged = {}
    if isinstance(source_component, dict):
        merged.update(source_component)
    merged.update(component)

    x = parse_float_maybe(merged.get("x"))
    y = parse_float_maybe(merged.get("y"))
    nearby = []
    if x is not None and y is not None:
        origin = np.asarray([float(x), float(y)], dtype=np.float32)
        for item in BOARD_MODEL.get("components", []):
            other_ref = str(item.get("ref", "")).strip()
            if not other_ref or other_ref == ref:
                continue
            other_x = parse_float_maybe(item.get("x"))
            other_y = parse_float_maybe(item.get("y"))
            if other_x is None or other_y is None:
                continue
            distance_mm = float(np.linalg.norm(origin - np.asarray([float(other_x), float(other_y)], dtype=np.float32)))
            nearby.append(
                {
                    "ref": other_ref,
                    "name": str(item.get("name") or "").strip(),
                    "device": str(item.get("device") or "").strip(),
                    "comment": str(item.get("comment") or "").strip(),
                    "footprint": str(item.get("footprint") or "").strip(),
                    "distanceMm": round(distance_mm, 3),
                }
            )
    nearby.sort(key=lambda item: (float(item.get("distanceMm", 9999.0)), item.get("ref", "")))
    nearby = nearby[: max(1, int(neighbor_limit))]

    nets = component_net_info(merged)
    net_summaries = []
    board_nets = BOARD_MODEL.get("nets", {})
    for item in nets[: max(1, int(net_limit))]:
        net_name = str(item.get("net") or "--").strip()
        peers = []
        payload = board_nets.get(net_name) if isinstance(board_nets, dict) else None
        nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        for node in nodes:
            peer_ref = str(node.get("ref", "")).strip()
            if not peer_ref or peer_ref == ref:
                continue
            peers.append(peer_ref)
        peers = sorted(dict.fromkeys(peers))[:4]
        net_summaries.append(
            {
                "net": net_name,
                "pin": str(item.get("pin") or "--").strip(),
                "name": str(item.get("name") or "--").strip(),
                "peerRefs": peers,
            }
        )

    name_text = str(merged.get("name") or "").strip()
    device_text = str(merged.get("device") or "").strip()
    comment_text = str(merged.get("comment") or "").strip()
    footprint_text = str(merged.get("footprint") or "").strip()
    nearby_text = "; ".join(
        f"{item.get('ref', '--')}({item.get('distanceMm', '--')}mm)"
        for item in nearby[:4]
    ) or "--"
    net_text = "; ".join(
        f"{item.get('net', '--')}/{item.get('pin', '--')}"
        + (f"-> {','.join(item.get('peerRefs', []))}" if item.get("peerRefs") else "")
        for item in net_summaries[:4]
    ) or "--"
    identity_parts = [text for text in (name_text, device_text, comment_text) if text]
    identity_text = " / ".join(dict.fromkeys(identity_parts)) if identity_parts else "--"

    return {
        "ref": ref,
        "name": name_text or "--",
        "device": device_text or "--",
        "comment": comment_text or "--",
        "footprint": footprint_text or "--",
        "layer": str(merged.get("layer") or "--").strip(),
        "pins": int(parse_float_maybe(merged.get("pins")) or 0),
        "smd": str(merged.get("smd") or "").strip() or "--",
        "boardPointMm": None if x is None or y is None else [round(float(x), 3), round(float(y), 3)],
        "nets": net_summaries,
        "nearby": nearby,
        "identityText": identity_text,
        "netText": net_text,
        "nearbyText": nearby_text,
        "summaryText": (
            f"位号 {ref}; 标识 {identity_text}; 封装 {footprint_text or '--'}; "
            f"网络 {net_text}; 邻近器件 {nearby_text}"
        ),
    }


def component_body_size_mm(component):
    ref = str(component.get("ref", "")).upper()
    footprint = str(component.get("footprint", "")).upper()
    device = str(component.get("device", "")).upper()
    comment = str(component.get("comment", "")).upper()

    footprint_dims = re.search(r"(?:^|[_-])L(?P<length>\d+(?:\.\d+)?)-W(?P<width>\d+(?:\.\d+)?)", footprint)
    if footprint_dims:
        try:
            length = float(footprint_dims.group("length"))
            width = float(footprint_dims.group("width"))
            if 0.6 <= length <= 80.0 and 0.6 <= width <= 80.0:
                return length, width
        except Exception:
            pass

    if "OLED" in footprint or "ST7735" in footprint or "LCD" in footprint or "DISPLAY" in device or "DISPLAY" in comment:
        return 22.0, 16.0
    if footprint.startswith("HDR-TH") or ref.startswith("H"):
        return 13.0, 5.0
    if footprint.startswith("R0603") or footprint == "R0603":
        return 1.8, 0.95
    if footprint.startswith("C0603") or footprint == "C0603":
        return 1.8, 0.95
    if "DFN-8" in footprint:
        return 3.2, 3.2
    if "QFP" in footprint:
        return 9.0, 9.0
    if "SOT-89" in footprint:
        return 4.8, 3.0
    if "CAP-SMD_L3.2-W1.6" in footprint:
        return 3.4, 1.8
    if ref.startswith("U"):
        return 5.0, 5.0
    if ref.startswith("C") or ref.startswith("R") or ref.startswith("D"):
        return 2.0, 1.2
    return 3.0, 3.0


COMPONENT_PROJECTION_CACHE = {
    "signature": None,
    "entries": None,
}


def component_corners_board(component, size_scale=1.0):
    try:
        cx = float(component["x"])
        cy = float(component["y"])
    except Exception:
        return None
    width, height = component_body_size_mm(component)
    width = max(0.6, float(width) * float(size_scale))
    height = max(0.6, float(height) * float(size_scale))
    theta = np.deg2rad(float(component.get("rotation", 0.0) or 0.0))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    half = np.array(
        [
            [-width / 2.0, -height / 2.0],
            [width / 2.0, -height / 2.0],
            [width / 2.0, height / 2.0],
            [-width / 2.0, height / 2.0],
        ],
        dtype=np.float32,
    )
    center = np.array([cx, cy], dtype=np.float32)
    return (half @ rot.T) + center


def build_component_projection_entries(board_matrix, visible_shape):
    if board_matrix is None or visible_shape is None:
        return []
    visible_h, visible_w = visible_shape[:2]
    signature = (
        BOARD_MODEL.get("id"),
        int(visible_h),
        int(visible_w),
        tuple(np.round(np.asarray(board_matrix, dtype=np.float32).reshape(-1), 3).tolist()),
    )
    if COMPONENT_PROJECTION_CACHE["entries"] is not None and COMPONENT_PROJECTION_CACHE["signature"] == signature:
        return COMPONENT_PROJECTION_CACHE["entries"]

    entries = []
    for component in BOARD_MODEL.get("components", []):
        try:
            if not is_visible_side_component(component):
                continue
            polygon_board = component_corners_board(component, size_scale=1.0)
            if polygon_board is None:
                continue
            polygon_visible = board_points_to_image(board_matrix, polygon_board)
            if polygon_visible.shape != (4, 2):
                continue

            x_min = float(np.min(polygon_visible[:, 0]))
            x_max = float(np.max(polygon_visible[:, 0]))
            y_min = float(np.min(polygon_visible[:, 1]))
            y_max = float(np.max(polygon_visible[:, 1]))
            if x_max < 0 or y_max < 0 or x_min >= visible_w or y_min >= visible_h:
                continue

            roi_x0 = max(0, int(math.floor(x_min)))
            roi_y0 = max(0, int(math.floor(y_min)))
            roi_x1 = min(visible_w, int(math.ceil(x_max)) + 1)
            roi_y1 = min(visible_h, int(math.ceil(y_max)) + 1)
            if roi_x1 <= roi_x0 + 1 or roi_y1 <= roi_y0 + 1:
                continue

            local_polygon = np.round(
                polygon_visible - np.array([roi_x0, roi_y0], dtype=np.float32)
            ).astype(np.int32)
            roi_mask = np.zeros((roi_y1 - roi_y0, roi_x1 - roi_x0), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [local_polygon], 255)
            mask_bool = roi_mask > 0
            pixel_count = int(np.count_nonzero(mask_bool))
            if pixel_count < 64:
                continue

            body_w, body_h = component_body_size_mm(component)
            entries.append(
                {
                    "component": component,
                    "mask": mask_bool,
                    "pixelCount": pixel_count,
                    "roi": (roi_x0, roi_y0, roi_x1, roi_y1),
                    "imageBounds": [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                    "bodyAreaMm2": round(float(body_w * body_h), 3),
                    "boardCenter": np.asarray([float(component["x"]), float(component["y"])], dtype=np.float32),
                }
            )
        except Exception:
            continue

    COMPONENT_PROJECTION_CACHE["signature"] = signature
    COMPONENT_PROJECTION_CACHE["entries"] = entries
    return entries


def component_distance_to_region_mm(component, board_point):
    if board_point is None:
        return None, None, None
    point = np.asarray(board_point, dtype=np.float32)
    polygon = component_corners_board(component, size_scale=1.0)
    expanded = component_corners_board(component, size_scale=1.35)
    if polygon is None or expanded is None:
        return None, None, None
    signed = float(cv2.pointPolygonTest(polygon.astype(np.float32), (float(point[0]), float(point[1])), True))
    expanded_signed = float(cv2.pointPolygonTest(expanded.astype(np.float32), (float(point[0]), float(point[1])), True))
    if signed >= 0:
        distance = 0.0
    else:
        distance = abs(signed)
    return distance, signed, expanded_signed


def nearest_component(board_point):
    if board_point is None:
        return None
    point = np.asarray(board_point, dtype=np.float32)
    best = None
    best_rank = None
    for component in BOARD_MODEL.get("components", []):
        if not is_visible_side_component(component):
            continue
        try:
            comp_point = np.array([float(component["x"]), float(component["y"])], dtype=np.float32)
        except Exception:
            continue
        distance = float(np.linalg.norm(comp_point - point))
        region_distance, signed_distance, expanded_signed = component_distance_to_region_mm(component, point)
        if region_distance is None:
            region_distance = distance
            signed_distance = -distance
            expanded_signed = -distance
        inside = 1 if signed_distance >= 0 else 0
        near_inside = 1 if expanded_signed >= 0 else 0
        width, height = component_body_size_mm(component)
        body_area = float(width * height)
        rank = (
            inside,
            near_inside,
            -region_distance,
            body_area,
            -distance,
        )
        if best_rank is None or rank > best_rank:
            best = dict(component)
            best_rank = rank
            best["distanceMm"] = round(distance, 3)
            best["regionDistanceMm"] = round(float(region_distance), 3)
            best["insideBody"] = bool(inside)
            best["nearBody"] = bool(near_inside)
            best["bodySizeMm"] = [round(width, 3), round(height, 3)]
    if best is None:
        return None
    best["nets"] = component_net_info(best)
    return best


def project_thermal_heat_to_visible(thermal_frame, thermal_to_visible_matrix, visible_shape):
    if thermal_frame is None or thermal_to_visible_matrix is None or visible_shape is None:
        return None
    heat = (
        thermal_frame[:, :, 2].astype(np.float32)
        + 0.5 * thermal_frame[:, :, 1].astype(np.float32)
        - 0.3 * thermal_frame[:, :, 0].astype(np.float32)
    )
    visible_h, visible_w = visible_shape[:2]
    return cv2.warpPerspective(heat, np.asarray(thermal_to_visible_matrix, dtype=np.float32), (visible_w, visible_h))


def project_thermal_mask_to_visible(mask, thermal_to_visible_matrix, visible_shape):
    if mask is None or thermal_to_visible_matrix is None or visible_shape is None:
        return None
    visible_h, visible_w = visible_shape[:2]
    warped = cv2.warpPerspective(
        mask.astype(np.uint8) * 255,
        np.asarray(thermal_to_visible_matrix, dtype=np.float32),
        (visible_w, visible_h),
        flags=cv2.INTER_NEAREST,
    )
    return warped > 0


def hottest_component_by_region(
    thermal_frame,
    thermal_to_visible_matrix,
    board_matrix,
    visible_shape,
    board_point=None,
    focus_projected_mask=None,
    prefer_overlap=False,
):
    if thermal_frame is None or thermal_to_visible_matrix is None or board_matrix is None or visible_shape is None:
        return None
    projected_heat = project_thermal_heat_to_visible(thermal_frame, thermal_to_visible_matrix, visible_shape)
    if projected_heat is None:
        return None

    projection_entries = build_component_projection_entries(board_matrix, visible_shape)
    if not projection_entries:
        return None

    focus_visible_point = None
    if board_point is not None:
        try:
            focus_visible_point = board_points_to_image(
                board_matrix,
                [np.asarray(board_point, dtype=np.float32)],
            )[0]
        except Exception:
            focus_visible_point = None

    candidates = []
    for entry in projection_entries:
        try:
            component = entry["component"]
            roi_x0, roi_y0, roi_x1, roi_y1 = entry["roi"]
            roi_heat = projected_heat[roi_y0:roi_y1, roi_x0:roi_x1]
            values = roi_heat[entry["mask"]]

            mean_heat = float(np.mean(values))
            sum_heat = float(np.sum(values))
            max_heat = float(np.max(values))
            center_distance = None
            region_distance = None
            inside_body = False
            near_body = False
            if board_point is not None:
                center_distance = float(
                    np.linalg.norm(
                        entry["boardCenter"] - np.asarray(board_point, dtype=np.float32)
                    )
                )
                region_distance, signed_distance, expanded_signed = component_distance_to_region_mm(
                    component,
                    board_point,
                )
                inside_body = bool(signed_distance is not None and signed_distance >= 0)
                near_body = bool(expanded_signed is not None and expanded_signed >= 0)

            local_sum_heat = sum_heat
            if focus_visible_point is not None:
                fx = int(round(float(focus_visible_point[0])))
                fy = int(round(float(focus_visible_point[1])))
                window_radius = 42
                wx0 = max(roi_x0, fx - window_radius)
                wy0 = max(roi_y0, fy - window_radius)
                wx1 = min(roi_x1, fx + window_radius + 1)
                wy1 = min(roi_y1, fy + window_radius + 1)
                if wx1 > wx0 and wy1 > wy0:
                    local_heat = projected_heat[wy0:wy1, wx0:wx1]
                    local_mask = entry["mask"][wy0 - roi_y0 : wy1 - roi_y0, wx0 - roi_x0 : wx1 - roi_x0]
                    if np.any(local_mask):
                        local_sum_heat = float(np.sum(local_heat[local_mask]))

            overlap_pixels = 0
            overlap_heat = 0.0
            if focus_projected_mask is not None:
                roi_focus = focus_projected_mask[roi_y0:roi_y1, roi_x0:roi_x1]
                if roi_focus.shape == entry["mask"].shape:
                    overlap_mask = roi_focus & entry["mask"]
                    if np.any(overlap_mask):
                        overlap_pixels = int(np.count_nonzero(overlap_mask))
                        overlap_heat = float(np.sum(roi_heat[overlap_mask]))

            candidates.append(
                {
                    "component": dict(component),
                    "meanHeat": mean_heat,
                    "sumHeat": sum_heat,
                    "localSumHeat": local_sum_heat,
                    "maxHeat": max_heat,
                    "pixelCount": entry["pixelCount"],
                    "bodyAreaMm2": entry["bodyAreaMm2"],
                    "centerDistanceMm": None if center_distance is None else round(center_distance, 3),
                    "regionDistanceMm": None if region_distance is None else round(float(region_distance), 3),
                    "insideBody": inside_body,
                    "nearBody": near_body,
                    "overlapPixels": overlap_pixels,
                    "overlapHeat": overlap_heat,
                    "imageBounds": entry["imageBounds"],
                }
            )
        except Exception:
            continue

    if not candidates:
        return None

    has_overlap_candidate = prefer_overlap and any(item["overlapPixels"] > 0 for item in candidates)
    if has_overlap_candidate:
        candidates.sort(
            key=lambda item: (
                1 if item["overlapPixels"] > 0 else 0,
                item["overlapHeat"],
                item["overlapPixels"],
                item["localSumHeat"],
                item["maxHeat"],
                1 if item["insideBody"] else 0,
                1 if item["nearBody"] else 0,
                -float(item["regionDistanceMm"] if item["regionDistanceMm"] is not None else 9999.0),
                item["meanHeat"],
                -float(item["centerDistanceMm"] or 9999.0),
                item["sumHeat"],
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (
                1 if item["insideBody"] else 0,
                1 if item["nearBody"] else 0,
                -float(item["regionDistanceMm"] if item["regionDistanceMm"] is not None else 9999.0),
                item["localSumHeat"],
                item["maxHeat"],
                item["meanHeat"],
                -float(item["centerDistanceMm"] or 9999.0),
                item["sumHeat"],
            ),
            reverse=True,
        )
    best = candidates[0]
    if board_point is not None and not best["nearBody"]:
        return None
    result = dict(best["component"])
    result["heatScore"] = round(best["sumHeat"], 2)
    result["localHeatScore"] = round(best["localSumHeat"], 2)
    result["meanHeat"] = round(best["meanHeat"], 2)
    result["maxHeat"] = round(best["maxHeat"], 2)
    result["pixelCount"] = best["pixelCount"]
    result["bodyAreaMm2"] = best["bodyAreaMm2"]
    result["centerDistanceMm"] = best["centerDistanceMm"]
    result["regionDistanceMm"] = best["regionDistanceMm"]
    result["insideBody"] = best["insideBody"]
    result["nearBody"] = best["nearBody"]
    result["overlapPixels"] = best["overlapPixels"]
    result["overlapHeat"] = round(best["overlapHeat"], 2)
    result["imageBounds"] = best["imageBounds"]
    result["distanceMm"] = best["centerDistanceMm"] if best["centerDistanceMm"] is not None else result.get("distanceMm")
    result["nets"] = component_net_info(result)
    result["matchMode"] = "projected-overlap-region" if has_overlap_candidate else "projected-heat-region"
    result["candidateSummary"] = [
        {
            "ref": item["component"].get("ref"),
            "insideBody": item["insideBody"],
            "nearBody": item["nearBody"],
            "regionDistanceMm": item["regionDistanceMm"],
            "overlapPixels": item["overlapPixels"],
            "overlapHeat": round(item["overlapHeat"], 2),
            "localSumHeat": round(item["localSumHeat"], 2),
            "sumHeat": round(item["sumHeat"], 2),
            "meanHeat": round(item["meanHeat"], 2),
            "maxHeat": round(item["maxHeat"], 2),
        }
        for item in candidates[:5]
    ]
    return result


def build_fault_probabilities(component, hotspot_score, hotspot_area, hotspot_bbox):
    ref = str(component.get("ref", "")).upper()
    footprint = str(component.get("footprint", "")).upper()
    comment = str(component.get("comment", "")).upper()
    device = str(component.get("device", "")).upper()
    distance = float(component.get("distanceMm", 0.0))
    area_factor = min(max(hotspot_area / 320.0, 0.0), 1.0)
    distance_factor = max(0.0, 1.0 - min(distance / 8.0, 1.0))
    hotspot_factor = min(max(hotspot_score / 50000.0, 0.0), 1.0)
    severity = 0.45 * hotspot_factor + 0.35 * distance_factor + 0.20 * area_factor

    if ref.startswith("C") or "CAP" in footprint or "UF" in comment:
        primary = ("电容异常发热/击穿", 0.48 + 0.34 * severity)
        secondary = ("周边供电纹波导致正常偏热", 0.18 + 0.10 * (1.0 - severity))
    elif ref.startswith("U") or "IC" in footprint or "RK" in device:
        primary = ("芯片负载过高或内部短路", 0.44 + 0.32 * severity)
        secondary = ("正常高负载工作发热", 0.20 + 0.12 * (1.0 - severity))
    elif ref.startswith("L"):
        primary = ("电感饱和或大电流发热", 0.46 + 0.28 * severity)
        secondary = ("电源环路正常温升", 0.24 + 0.10 * (1.0 - severity))
    elif ref.startswith("Q") or ref.startswith("D"):
        primary = ("功率器件漏电或开关损耗偏高", 0.47 + 0.28 * severity)
        secondary = ("开关工况正常发热", 0.22 + 0.08 * (1.0 - severity))
    else:
        primary = ("器件本体异常发热", 0.40 + 0.28 * severity)
        secondary = ("邻近热源传导导致偏热", 0.26 + 0.12 * (1.0 - severity))

    tertiary = ("当前工况正常温升", max(0.08, 1.0 - primary[1] - secondary[1]))
    raw = [primary, secondary, tertiary]
    total = sum(max(prob, 0.01) for _, prob in raw)
    probabilities = []
    remaining = 100
    for index, (label, prob) in enumerate(raw):
        percent = int(round(max(prob, 0.01) / total * 100))
        if index == len(raw) - 1:
            percent = remaining
        remaining -= percent
        probabilities.append({"label": label, "percent": max(percent, 0)})
    probabilities.sort(key=lambda item: item["percent"], reverse=True)
    return probabilities


def call_deepseek_summary(report_payload):
    config = load_llm_config()
    if not config.get("enabled"):
        return {"ok": False, "error": "deepseek api key not configured", "provider": "deepseek"}

    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    nets = nearest.get("nets") or []
    candidate_summary = nearest.get("candidateSummary") or []
    nets_text = "\n".join(
        f"- 网络：{item.get('net', '--')}，引脚：{item.get('pin', '--')}，名称：{item.get('name', '--')}"
        for item in nets[:8]
    ) or "- 暂无网表信息"
    candidate_text = "\n".join(
        f"- {item.get('ref', '--')} / sumHeat={item.get('sumHeat', '--')} / meanHeat={item.get('meanHeat', '--')}"
        for item in candidate_summary[:5]
    ) or "- 暂无候选器件热量排名"
    source_file = BOARD_MODEL.get("sourceFile") or "default-board-model"
    prompt = (
        "你是PCB热诊断工程师。请基于以下结构化信息，输出详细、专业、全中文的诊断结果。"
        "要求：\n"
        "1. 用JSON格式返回，不要输出Markdown。\n"
        "2. JSON字段必须包含 summary、riskLevel、actions、detailedAnalysis、probabilities、componentProfile。\n"
        "3. summary 控制在80字以内，适合首页摘要。\n"
        "4. detailedAnalysis 是1到2段更详细中文分析。\n"
        "5. probabilities 是数组，每项包含 label 和 percent，总和尽量为100。\n"
        "6. componentProfile 是对象，包含 probableFunction、packageNotes、datasheetHints 三个字段。\n"
        "7. actions 为中文字符串数组，给3条以内建议。\n"
        "8. 允许根据器件型号、封装、常见用途、网表关系做合理推断，但要避免虚构绝对结论。\n"
        "9. 如果器件型号看起来像可检索器件，可基于常见公开资料给出用途或参数方向，但不要伪造精确未确认数值。\n\n"
        f"板卡编号：{report_payload.get('boardId', '--')}\n"
        f"坐标来源文件：{source_file}\n"
        f"生成时间：{report_payload.get('generatedAt', '--')}\n"
        f"热源定位来源：{analysis.get('hotspotSource', '--')}\n"
        f"热像准星点：{analysis.get('thermalPoint', '--')}\n"
        f"映射后的可见光点：{analysis.get('imagePoint', '--')}\n"
        f"PCB坐标(mm)：{analysis.get('boardPointMm', '--')}\n"
        f"最近器件：{nearest.get('ref', '--')}\n"
        f"封装：{nearest.get('footprint', '--')}\n"
        f"型号：{nearest.get('device', '--')}\n"
        f"说明：{nearest.get('comment', '--')}\n"
        f"距离(mm)：{nearest.get('distanceMm', '--')}\n"
        f"匹配模式：{nearest.get('matchMode', '--')}\n"
        f"投影区域热量分数：{nearest.get('heatScore', '--')}\n"
        f"候选器件热量排名：\n{candidate_text}\n"
        "网表信息：\n"
        f"{nets_text}\n"
    )
    payload = {
        "model": config.get("model") or DEEPSEEK_DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一名严谨的PCB故障诊断助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    url = f"{config.get('baseUrl', DEEPSEEK_BASE_URL)}{DEEPSEEK_CHAT_PATH}"
    response = http_post_json_headers(
        url,
        payload,
        headers={"Authorization": f"Bearer {config['apiKey']}"},
        timeout=35,
    )
    message = (((response.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not message:
        return {"ok": False, "error": "deepseek returned empty content", "provider": "deepseek", "raw": response}
    try:
        parsed = json.loads(message)
    except Exception:
        parsed = {
            "summary": str(message).strip(),
            "riskLevel": "中",
            "actions": [],
            "detailedAnalysis": str(message).strip(),
            "probabilities": [],
            "componentProfile": {},
        }
    parsed["ok"] = True
    parsed["provider"] = "deepseek"
    parsed["model"] = payload["model"]
    return parsed


def detect_thermal_crosshair_point(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv, np.array([0, 110, 110], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    red_b = cv2.inRange(hsv, np.array([165, 110, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(red_a, red_b)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if int(np.count_nonzero(red)) < 12:
        return None

    binary = (red > 0).astype(np.uint8)
    height, width = binary.shape
    best = None
    best_score = 0.0
    best_h = 0
    best_v = 0
    ys_nonzero, xs_nonzero = np.where(binary > 0)
    if len(xs_nonzero) == 0:
        return None
    x_min = int(xs_nonzero.min())
    x_max = int(xs_nonzero.max())
    y_min = int(ys_nonzero.min())
    y_max = int(ys_nonzero.max())
    search_top = y_min
    search_bottom = min(y_max, y_min + max(24, int((y_max - y_min) * 0.42)))

    def directional_runs(mask, x, y):
        left = 0
        right = 0
        up = 0
        down = 0
        cursor = x - 1
        while cursor >= 0 and mask[y, cursor]:
            left += 1
            cursor -= 1
        cursor = x + 1
        while cursor < width and mask[y, cursor]:
            right += 1
            cursor += 1
        cursor = y - 1
        while cursor >= 0 and mask[cursor, x]:
            up += 1
            cursor -= 1
        cursor = y + 1
        while cursor < height and mask[cursor, x]:
            down += 1
            cursor += 1
        return left, right, up, down

    ys, xs = np.where(binary > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if y < search_top or y > search_bottom:
            continue
        left, right, up, down = directional_runs(binary, x, y)
        h_run = left + right + 1
        v_run = up + down + 1
        if min(left, right) < 4 or min(up, down) < 3:
            continue
        patch = frame[max(0, y - 6) : min(height, y + 7), max(0, x - 6) : min(width, x + 7)]
        warmth = 0.0
        if patch.size:
            warmth = float(np.mean(patch[:, :, 2]) + 0.30 * np.mean(patch[:, :, 1]) - 0.20 * np.mean(patch[:, :, 0]))
        symmetry = min(left, right) + min(up, down)
        balance_penalty = abs(left - right) + abs(up - down)
        vertical_bonus = max(0, search_bottom - y) * 1.4
        score = float(h_run * max(v_run, 6)) + warmth * 1.15 + symmetry * 18.0 + vertical_bonus - balance_penalty * 6.0
        if score > best_score:
            best_score = score
            best = (x, y)
            best_h = h_run
            best_v = v_run

    if best is None:
        return None

    x, y = best
    arm = max(12, int(round(max(best_h, best_v) * 0.75)))
    x0 = max(0, x - arm)
    y0 = max(0, y - arm)
    x1 = min(width, x + arm + 1)
    y1 = min(height, y + arm + 1)
    patch = frame[y0:y1, x0:x1]
    warmth = 0.0
    if patch.size:
        warmth = float(np.mean(patch[:, :, 2]) + 0.35 * np.mean(patch[:, :, 1]) - 0.25 * np.mean(patch[:, :, 0]))
    return {
        "center": np.array([float(x), float(y)], dtype=np.float32),
        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "area": float(best_h * best_v),
        "score": float(best_score),
        "warmth": warmth,
        "source": "thermal-crosshair",
    }


def match_hotspot_with_obb(visible_point, detections):
    if visible_point is None or not detections:
        return None
    point = np.asarray(visible_point, dtype=np.float32)
    best = None
    best_distance = None
    for det in detections:
        quad = np.asarray(det.get("quad", []), dtype=np.float32)
        if quad.shape != (4, 2):
            continue
        center = np.mean(quad, axis=0)
        distance = float(np.linalg.norm(center - point))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = {
                "label": det.get("label"),
                "score": round(float(det.get("score", 0.0) or 0.0), 4),
                "distancePx": round(distance, 2),
                "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
                "quad": np.round(quad, 2).tolist(),
            }
    return best


def analyze_hotspot(
    thermal,
    adjusted_matrix,
    board_matrix=None,
    visible_quad=None,
    thermal_quad=None,
    manual_adjust=None,
    yolo_detections=None,
    visible_shape=None,
    enable_ocr=True,
    cached_thermal_ocr=None,
):
    crosshair = detect_thermal_crosshair_point(thermal)
    region = None
    focus_projected_mask = None
    prefer_overlap = False
    if crosshair is not None:
        region = refine_crosshair_hotspot_target(thermal, crosshair["center"])
        if region is None:
            region = crosshair
    if region is None:
        region = detect_thermal_hotspot_region(thermal)
    if region is None:
        return None
    hotspot_center = np.asarray(region.get("center"), dtype=np.float32)
    target_center = hotspot_center
    if crosshair is not None and crosshair.get("center") is not None:
        try:
            target_center = np.asarray(crosshair.get("center"), dtype=np.float32)
        except Exception:
            target_center = hotspot_center
    association_center = hotspot_center
    if float(np.linalg.norm(hotspot_center - target_center)) > 48.0:
        association_center = target_center
    mapped = cv2.perspectiveTransform(
        np.array([[association_center]], dtype=np.float32),
        np.asarray(adjusted_matrix, dtype=np.float32),
    )[0][0]
    crosshair_mapped = cv2.perspectiveTransform(
        np.array([[target_center]], dtype=np.float32),
        np.asarray(adjusted_matrix, dtype=np.float32),
    )[0][0]
    board_point = None
    nearest = None
    obb_match = match_hotspot_with_obb(mapped, yolo_detections or [])
    if board_matrix is not None:
        board_point = image_points_to_board(board_matrix, [mapped])[0]
        if crosshair is not None and visible_shape is not None:
            local_mask = build_crosshair_guided_hot_mask(thermal, crosshair["center"])
            if local_mask is not None:
                focus_projected_mask = project_thermal_mask_to_visible(local_mask, adjusted_matrix, visible_shape)
                if focus_projected_mask is not None:
                    focus_projected_mask = cv2.dilate(
                        focus_projected_mask.astype(np.uint8) * 255,
                        np.ones((7, 7), np.uint8),
                        iterations=1,
                    ) > 0
                    prefer_overlap = True
        nearest = hottest_component_by_region(
            thermal,
            adjusted_matrix,
            board_matrix,
            visible_shape,
            board_point=board_point,
            focus_projected_mask=focus_projected_mask,
            prefer_overlap=prefer_overlap,
        )
        if nearest is None:
            fallback = nearest_component(board_point)
            if fallback is not None and (
                fallback.get("insideBody")
                or fallback.get("nearBody")
                or float(fallback.get("distanceMm", 9999.0) or 9999.0) <= 6.0
            ):
                fallback["matchMode"] = "nearest-component-fallback"
                nearest = fallback
        if nearest is None and obb_match is not None and obb_match.get("center") is not None:
            try:
                obb_board_point = image_points_to_board(
                    board_matrix,
                    [np.asarray(obb_match["center"], dtype=np.float32)],
                )[0]
                obb_fallback = nearest_component(obb_board_point)
                if (
                    obb_fallback is not None
                    and float(obb_match.get("distancePx", 9999.0) or 9999.0) <= 18.0
                    and float(obb_fallback.get("distanceMm", 9999.0) or 9999.0) <= 6.0
                ):
                    obb_fallback["matchMode"] = "obb-nearest-component"
                    obb_fallback["obbDistancePx"] = round(float(obb_match.get("distancePx", 0.0) or 0.0), 2)
                    nearest = obb_fallback
            except Exception:
                pass
        if nearest is not None:
            nearest["imagePoint"] = [round(float(v), 2) for v in board_points_to_image(board_matrix, [[nearest["x"], nearest["y"]]])[0]]
    bbox = region.get("bbox", [0, 0, 0, 0])
    probabilities = build_fault_probabilities(nearest, region["score"], region["area"], bbox) if nearest is not None else []
    thermal_ocr = cached_thermal_ocr
    if enable_ocr:
        thermal_ocr = extract_thermal_temperature_ocr(thermal, focus_point=target_center)
    return {
        "imagePoint": [round(float(mapped[0]), 2), round(float(mapped[1]), 2)],
        "crosshairImagePoint": [round(float(crosshair_mapped[0]), 2), round(float(crosshair_mapped[1]), 2)],
        "thermalPoint": [round(float(association_center[0]), 2), round(float(association_center[1]), 2)],
        "thermalPeakPoint": [round(float(hotspot_center[0]), 2), round(float(hotspot_center[1]), 2)],
        "crosshairPoint": region.get("crosshairPoint"),
        "aimPoint": region.get("aimPoint"),
        "boardPointMm": None if board_point is None else [round(float(board_point[0]), 3), round(float(board_point[1]), 3)],
        "bboxThermal": [round(float(v), 2) for v in bbox],
        "areaPx": round(float(region["area"]), 2),
        "score": round(float(region["score"]), 2),
        "warmth": round(float(region["warmth"]), 2),
        "hotspotSource": str(region.get("source", "thermal-region")),
        "nearestComponent": nearest,
        "obbMatch": obb_match,
        "probabilities": probabilities,
        "thermalOcr": thermal_ocr,
        "thermalTemperatureC": None if not thermal_ocr or not thermal_ocr.get("ok") else thermal_ocr.get("temperatureC"),
        "visibleQuad": None if visible_quad is None else np.round(np.asarray(visible_quad), 2).tolist(),
        "thermalQuad": None if thermal_quad is None else np.round(np.asarray(thermal_quad), 2).tolist(),
        "manualAdjust": manual_adjust or default_manual_adjust(),
    }


def analyze_hotspot_from_frames(visible, thermal, matrix, rig_matrix, visible_quad, thermal_quad, board_matrix, manual_adjust, yolo_detections=None):
    if visible is None or thermal is None:
        return None
    adjusted_visible_quad = visible_quad
    if visible_quad is None:
        adjusted_visible_quad = None
    elif manual_adjust:
        adjusted_visible_quad = adjust_quad(visible_quad, manual_adjust)

    if board_matrix is None and visible_quad is not None:
        board_matrix = board_matrix_from_quad(visible_quad)

    if visible_quad is not None and thermal_quad is not None:
        adjusted_thermal_quad = adjust_source_quad(thermal_quad, manual_adjust, thermal.shape)
        if matrix is not None:
            adjusted_matrix = compose_board_space_matrix(matrix, visible_quad, adjusted_visible_quad)
        elif rig_matrix is not None:
            adjusted_matrix = compose_board_space_matrix(rig_matrix, visible_quad, adjusted_visible_quad)
        else:
            adjusted_matrix = cv2.getPerspectiveTransform(adjusted_thermal_quad.astype(np.float32), adjusted_visible_quad.astype(np.float32))
    else:
        source = rig_matrix if rig_matrix is not None else matrix
        if source is None:
            return None
        adjusted_matrix = np.asarray(source, dtype=np.float32)

    return analyze_hotspot(
        thermal,
        adjusted_matrix,
        board_matrix=board_matrix,
        visible_quad=visible_quad,
        thermal_quad=thermal_quad,
        manual_adjust=manual_adjust,
        yolo_detections=yolo_detections,
        visible_shape=visible.shape,
    )


def draw_board_model_overlay(frame, board_matrix):
    corners = board_points_to_image(board_matrix, board_corners_mm()).astype(np.int32)
    cv2.polylines(frame, [corners], True, (0, 255, 80), 2)

    holes = BOARD_MODEL.get("holeCentersMm", {})
    for ref, point in holes.items():
        mapped = board_points_to_image(board_matrix, [[float(point["x"]), float(point["y"])]])[0]
        draw_text_with_backing(frame, ref, mapped + np.array([8, -8], dtype=np.float32), (255, 180, 80), 0.42, 1)

    label_refs = set(BOARD_MODEL.get("labelRefs", []))
    components = BOARD_MODEL.get("components", [])
    for component in components:
        ref = str(component.get("ref", "")).strip()
        if not ref or ref not in label_refs:
            continue
        if not is_visible_side_component(component):
            continue
        mapped = board_points_to_image(board_matrix, [component_label_board_point(component)])[0]
        x, y = mapped
        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 230, 255), -1, cv2.LINE_AA)
        draw_text_with_backing(frame, ref, mapped + np.array([4, -4], dtype=np.float32), (0, 255, 255), 0.34, 1)


def build_hole_debug_frame(visible, visible_quad, board_matrix, hole_debug):
    if visible is None:
        return None
    frame = visible.copy()
    if visible_quad is not None:
        cv2.polylines(frame, [np.asarray(visible_quad, dtype=np.int32)], True, (0, 255, 120), 3)
    if board_matrix is not None:
        corners = board_points_to_image(board_matrix, board_corners_mm()).astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 210, 255), 2)
        for ref, point in BOARD_MODEL.get("holeCentersMm", {}).items():
            mapped = board_points_to_image(board_matrix, [[float(point["x"]), float(point["y"])]])[0]
            draw_text_with_backing(frame, ref, mapped + np.array([8, -8], dtype=np.float32), (255, 180, 80), 0.45, 1)

    for item in (hole_debug or {}).get("candidates", []):
        predicted = item.get("predicted")
        if predicted:
            p = np.asarray(predicted, dtype=np.float32)
            cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 14, (255, 0, 255), 2, cv2.LINE_AA)
        best = item.get("best")
        if best and best.get("point"):
            p = np.asarray(best["point"], dtype=np.float32)
            cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 20, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{item.get('ref')} {best.get('score')}",
                (int(round(p[0] + 12)), int(round(p[1] + 18))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    label = f"hole debug: {(hole_debug or {}).get('method', 'none')} / {(hole_debug or {}).get('reason', 'none')}"
    if (hole_debug or {}).get("accepted"):
        label += f" / score {(hole_debug or {}).get('score')} / err {(hole_debug or {}).get('errorPx')}px"
    draw_label(frame, label, (24, 46), (255, 255, 255))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return None
    return encoded.tobytes()


def fit_rect(src_w, src_h, dst_w, dst_h):
    scale = min(dst_w / src_w, dst_h / src_h)
    w = max(1, int(src_w * scale))
    h = max(1, int(src_h * scale))
    x = (dst_w - w) // 2
    y = (dst_h - h) // 2
    return x, y, w, h


def paste_resized(dst, src, rect):
    x, y, w, h = rect
    resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_LANCZOS4)
    dst[y : y + h, x : x + w] = resized


def alpha_blend_mask(base, overlay, mask, alpha):
    result = base.copy()
    valid = mask > 0
    if not np.any(valid):
        return result
    blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
    result[valid] = blended[valid]
    return result


def apply_cached_overlay(base, overlay):
    if overlay is None:
        return base
    valid = np.any(overlay > 0, axis=2)
    if np.any(valid):
        base[valid] = overlay[valid]
    return base


def adjust_quad(quad, adjust):
    adjusted = np.asarray(quad, dtype=np.float32).copy()
    center = adjusted.mean(axis=0)
    scale = np.array([adjust.get("scaleX", 1.0), adjust.get("scaleY", 1.0)], dtype=np.float32)
    offset = np.array([adjust.get("dx", 0.0), adjust.get("dy", 0.0)], dtype=np.float32)
    return (adjusted - center) * scale + center + offset


def adjust_source_quad(quad, adjust, frame_shape):
    adjusted = np.asarray(quad, dtype=np.float32)
    if adjusted.shape != (4, 2):
        return adjusted
    adjusted = adjusted.copy()
    height, width = frame_shape[:2]
    left = adjust.get("sourceLeft", 0.0)
    right = adjust.get("sourceRight", 0.0)
    top = adjust.get("sourceTop", 0.0)
    bottom = adjust.get("sourceBottom", 0.0)

    adjusted[0, 0] -= left
    adjusted[3, 0] -= left
    adjusted[1, 0] += right
    adjusted[2, 0] += right
    adjusted[0, 1] -= top
    adjusted[1, 1] -= top
    adjusted[2, 1] += bottom
    adjusted[3, 1] += bottom

    adjusted[:, 0] = np.clip(adjusted[:, 0], 0, width - 1)
    adjusted[:, 1] = np.clip(adjusted[:, 1], 0, height - 1)
    return adjusted


def compose_board_space_matrix(rig_matrix, original_visible_quad, adjusted_visible_quad):
    if rig_matrix is None:
        return None
    src = np.asarray(original_visible_quad, dtype=np.float32)
    dst = np.asarray(adjusted_visible_quad, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        return np.asarray(rig_matrix, dtype=np.float32)
    visible_adjust = cv2.getPerspectiveTransform(src, dst)
    return visible_adjust @ np.asarray(rig_matrix, dtype=np.float32)


def compose_board_matrix(board_matrix, thermal_board_matrix, original_visible_quad, adjusted_visible_quad):
    if board_matrix is None or thermal_board_matrix is None:
        return None
    src = np.asarray(original_visible_quad, dtype=np.float32)
    dst = np.asarray(adjusted_visible_quad, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        visible_adjust = np.eye(3, dtype=np.float32)
    else:
        visible_adjust = cv2.getPerspectiveTransform(src, dst)
    return visible_adjust @ np.asarray(board_matrix, dtype=np.float32) @ np.linalg.inv(np.asarray(thermal_board_matrix, dtype=np.float32))


def detect_thermal_hotspot_region(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, np.array([0, 80, 120], dtype=np.uint8), np.array([18, 255, 255], dtype=np.uint8))
    mask_b = cv2.inRange(hsv, np.array([18, 70, 150], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(mask_a, mask_b)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(frame_area * 0.0008, 6) or area > frame_area * 0.45:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
        x, y, w, h = cv2.boundingRect(contour)
        patch = frame[y : y + h, x : x + w]
        if patch.size == 0:
            continue
        warmth = float(np.mean(patch[:, :, 2]) + 0.35 * np.mean(patch[:, :, 1]) - 0.25 * np.mean(patch[:, :, 0]))
        candidates.append(
            {
                "area": float(area),
                "center": center,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(area * warmth),
                "contour": contour,
                "warmth": warmth,
            }
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["score"])
    best["mask"] = mask
    return best


def find_local_hot_peak(frame, center_point, min_radius=0, max_radius=42):
    if frame is None:
        return None
    height, width = frame.shape[:2]
    cx = int(round(float(center_point[0])))
    cy = int(round(float(center_point[1])))
    radius = int(max(min_radius, max_radius))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius + 1)
    y2 = min(height, cy + radius + 1)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None

    patch = frame[y1:y2, x1:x2]
    heat = patch[:, :, 2].astype(np.float32) + 0.5 * patch[:, :, 1].astype(np.float32) - 0.3 * patch[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(heat, (0, 0), 2.2)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv_patch, np.array([0, 110, 110], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)) > 0
    red_b = cv2.inRange(hsv_patch, np.array([165, 110, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)) > 0
    red_marks = red_a | red_b
    warm_colors = (
        ((hsv_patch[:, :, 0] <= 45) & (hsv_patch[:, :, 1] >= 40) & (hsv_patch[:, :, 2] >= 110))
        | (hsv_patch[:, :, 2] >= 235)
    )

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dx = xx - float(cx)
    dy = yy - float(cy)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ring = (dist >= float(min_radius)) & (dist <= float(max_radius))
    if not np.any(ring):
        return None

    valid = ring & (~red_marks)
    if np.any(valid & warm_colors):
        valid = valid & warm_colors
    elif not np.any(valid):
        valid = ring

    radial_penalty = dist * 0.38
    color_bonus = np.where(warm_colors, 20.0, -10.0)
    weighted = np.where(valid, blurred - radial_penalty + color_bonus, -1e9)
    flat_index = int(np.argmax(weighted))
    local_y, local_x = np.unravel_index(flat_index, weighted.shape)
    if weighted[local_y, local_x] < -1e8:
        return None

    peak_x = int(local_x + x1)
    peak_y = int(local_y + y1)
    peak_score = float(weighted[local_y, local_x])

    box = 12
    px1 = max(0, peak_x - box)
    py1 = max(0, peak_y - box)
    px2 = min(width, peak_x + box + 1)
    py2 = min(height, peak_y + box + 1)
    peak_patch = frame[py1:py2, px1:px2]
    warmth = 0.0
    if peak_patch.size:
        warmth = float(np.mean(peak_patch[:, :, 2]) + 0.35 * np.mean(peak_patch[:, :, 1]) - 0.25 * np.mean(peak_patch[:, :, 0]))
    return {
        "center": np.array([float(peak_x), float(peak_y)], dtype=np.float32),
        "bbox": [float(px1), float(py1), float(px2 - px1), float(py2 - py1)],
        "area": float((px2 - px1) * (py2 - py1)),
        "score": peak_score * float((px2 - px1) * (py2 - py1)),
        "warmth": warmth,
        "peakValue": peak_score,
        "source": "thermal-crosshair-peak",
    }


def build_crosshair_guided_hot_mask(frame, crosshair_point, max_radius=54):
    if frame is None or crosshair_point is None:
        return None
    height, width = frame.shape[:2]
    cx = float(crosshair_point[0])
    cy = float(crosshair_point[1])
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    local = dist <= float(max_radius)
    if not np.any(local):
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    heat = frame[:, :, 2].astype(np.float32) + 0.5 * frame[:, :, 1].astype(np.float32) - 0.3 * frame[:, :, 0].astype(np.float32)
    red_a = cv2.inRange(hsv, np.array([0, 110, 110], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)) > 0
    red_b = cv2.inRange(hsv, np.array([165, 110, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)) > 0
    red_marks = red_a | red_b
    warm_mask = (
        (((hsv[:, :, 0] <= 45) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 110)) | (hsv[:, :, 2] >= 235))
        & (~red_marks)
        & local
    )
    if not np.any(warm_mask):
        warm_mask = local & (~red_marks)
    if not np.any(warm_mask):
        return None

    values = heat[warm_mask]
    if values.size < 8:
        return None
    threshold = float(np.percentile(values, 72))
    hot_mask = warm_mask & (heat >= threshold)
    hot_mask = cv2.morphologyEx((hot_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    if int(np.count_nonzero(hot_mask)) < 6:
        return None
    return hot_mask > 0


def refine_crosshair_hotspot_target(frame, crosshair_point):
    crosshair = np.asarray(crosshair_point, dtype=np.float32)
    local_peak = find_local_hot_peak(frame, crosshair, min_radius=0, max_radius=42)
    if local_peak is not None:
        center = np.asarray(local_peak["center"], dtype=np.float32)
        if float(np.linalg.norm(center - crosshair)) <= 42.0:
            local_peak["aimPoint"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
            local_peak["crosshairPoint"] = [round(float(crosshair[0]), 2), round(float(crosshair[1]), 2)]
            local_peak["source"] = "thermal-crosshair-hotspot"
            return local_peak

    cx = int(round(float(crosshair[0])))
    cy = int(round(float(crosshair[1])))
    box = 12
    x1 = max(0, cx - box)
    y1 = max(0, cy - box)
    x2 = min(frame.shape[1], cx + box + 1)
    y2 = min(frame.shape[0], cy + box + 1)
    patch = frame[y1:y2, x1:x2]
    warmth = 0.0
    if patch.size:
        warmth = float(np.mean(patch[:, :, 2]) + 0.35 * np.mean(patch[:, :, 1]) - 0.25 * np.mean(patch[:, :, 0]))
    return {
        "center": np.array([float(cx), float(cy)], dtype=np.float32),
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "area": float((x2 - x1) * (y2 - y1)),
        "score": warmth * float(max(1, (x2 - x1) * (y2 - y1))),
        "warmth": warmth,
        "aimPoint": [round(float(cx), 2), round(float(cy), 2)],
        "crosshairPoint": [round(float(crosshair[0]), 2), round(float(crosshair[1]), 2)],
        "source": "thermal-crosshair-hotspot",
    }


def order_polygon_clockwise(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order]


def draw_obb_overlay(frame, detections, threshold=0.0):
    if not detections:
        return
    for index, det in enumerate(detections, start=1):
        score = det.get("score")
        if score is not None and float(score) < float(threshold):
            continue
        quad = np.asarray(det.get("quad", []), dtype=np.float32)
        if quad.shape != (4, 2):
            continue
        ordered = order_polygon_clockwise(quad).astype(np.int32)
        cv2.polylines(frame, [ordered], True, (255, 0, 255), 2, cv2.LINE_AA)
        center = np.mean(quad, axis=0)
        label = det.get("label") or f"obb-{index}"
        if score is not None:
            label = f"{label} {float(score):.2f}"
        draw_text_with_backing(frame, label, center + np.array([6, -6], dtype=np.float32), (255, 0, 255), 0.42, 1)


def letterbox_resize(image, size=(640, 640)):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = image_resized
    return canvas, scale, dx, dy


def build_board_crop(frame, quad, pad_ratio=0.05):
    if quad is None:
        return frame.copy(), (0, 0)
    pts = np.asarray(quad, dtype=np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    pad_x = width * pad_ratio
    pad_y = height * pad_ratio
    x1 = max(0, int(np.floor(min_xy[0] - pad_x)))
    y1 = max(0, int(np.floor(min_xy[1] - pad_y)))
    x2 = min(frame.shape[1], int(np.ceil(max_xy[0] + pad_x)))
    y2 = min(frame.shape[0], int(np.ceil(max_xy[1] + pad_y)))
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def clamp_rect_xyxy(rect, frame_shape):
    if rect is None or frame_shape is None:
        return None
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in rect]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def xywh_to_xyxy(rect):
    if rect is None:
        return None
    x, y, w, h = [float(v) for v in rect]
    return [x, y, x + w, y + h]


def union_rects(rects):
    valid = [rect for rect in rects if rect is not None]
    if not valid:
        return None
    x1 = min(float(rect[0]) for rect in valid)
    y1 = min(float(rect[1]) for rect in valid)
    x2 = max(float(rect[2]) for rect in valid)
    y2 = max(float(rect[3]) for rect in valid)
    return [x1, y1, x2, y2]


def quad_to_rect(quad):
    if quad is None:
        return None
    try:
        pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if pts.size == 0:
        return None
    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def expand_rect(rect, pad_x, pad_y, frame_shape):
    if rect is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in rect]
    return clamp_rect_xyxy([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], frame_shape)


def crop_frame_rect(frame, rect):
    if frame is None:
        return None, None
    rect = clamp_rect_xyxy(rect, frame.shape)
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    return frame[y1:y2, x1:x2].copy(), rect


def crop_board_region(frame, board_rect, pad_ratio=0.06):
    if frame is None or board_rect is None:
        return None, None
    x1, y1, x2, y2 = [float(v) for v in board_rect]
    pad_x = max(6.0, (x2 - x1) * float(pad_ratio))
    pad_y = max(6.0, (y2 - y1) * float(pad_ratio))
    return crop_frame_rect(frame, [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y])


def crop_focus_region(frame, center=None, focus_rect=None, min_size=None, aspect_ratio=1.55, pad_ratio=0.16):
    if frame is None:
        return None, None
    height, width = frame.shape[:2]
    min_w, min_h = min_size or (max(120, width // 5), max(90, height // 5))
    if focus_rect is not None:
        x1, y1, x2, y2 = [float(v) for v in focus_rect]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        target_w = max(min_w, int(round(box_w * (1.0 + pad_ratio * 2.0))))
        target_h = max(min_h, int(round(box_h * (1.0 + pad_ratio * 2.0))))
        if center is None:
            center = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
    else:
        target_w = min_w
        target_h = min_h
    if center is None:
        center = [width * 0.5, height * 0.5]

    target_w = max(target_w, int(round(target_h * aspect_ratio)))
    target_h = max(target_h, int(round(target_w / max(aspect_ratio, 1e-6))))
    target_w = min(target_w, width)
    target_h = min(target_h, height)

    cx, cy = float(center[0]), float(center[1])
    x1 = int(round(cx - target_w * 0.5))
    y1 = int(round(cy - target_h * 0.5))
    x1 = max(0, min(width - target_w, x1))
    y1 = max(0, min(height - target_h, y1))
    x2 = x1 + target_w
    y2 = y1 + target_h
    rect = [x1, y1, x2, y2]
    return frame[y1:y2, x1:x2].copy(), rect


def save_report_image(image, stem):
    if image is None or image.size == 0:
        return None
    output_path = reports_dir() / f"{stem}.jpg"
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_path


def decode_jpeg_bytes(payload):
    if not payload:
        return None
    try:
        array = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        return None


THERMAL_TEMPERATURE_RE = re.compile(r"(-?\d{1,3}(?:[.,]\d{1,2})?)")
THERMAL_CHAR_TEMPLATE_CACHE = None
RAPIDOCR_ENGINE = None
RAPIDOCR_AVAILABLE = None


def trim_binary_mask(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return mask
    x, y, w, h = cv2.boundingRect(coords)
    return mask[y : y + h, x : x + w]


def normalize_binary_mask(mask, width=28, height=40):
    trimmed = trim_binary_mask(mask)
    if trimmed is None or trimmed.size == 0 or int(np.count_nonzero(trimmed)) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    src_h, src_w = trimmed.shape[:2]
    scale = min((width - 4) / max(src_w, 1), (height - 4) / max(src_h, 1))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(trimmed, (out_w, out_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width), dtype=np.uint8)
    x0 = (width - out_w) // 2
    y0 = (height - out_h) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return canvas


def thermal_char_templates():
    global THERMAL_CHAR_TEMPLATE_CACHE
    if THERMAL_CHAR_TEMPLATE_CACHE is not None:
        return THERMAL_CHAR_TEMPLATE_CACHE
    templates = {}
    chars = "0123456789.-C"
    for char in chars:
        variants = []
        for scale in (0.62, 0.76, 0.9, 1.05):
            for thickness in (1, 2, 3):
                canvas = np.zeros((64, 48), dtype=np.uint8)
                cv2.putText(canvas, char, (3, 46), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_AA)
                normalized = normalize_binary_mask(canvas)
                if int(np.count_nonzero(normalized)) > 0:
                    variants.append(normalized)
        templates[char] = variants
    THERMAL_CHAR_TEMPLATE_CACHE = templates
    return templates


def ensure_binary_foreground(mask):
    if mask is None or mask.size == 0:
        return mask
    binary = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(binary)) > binary.size * 0.56:
        binary = 255 - binary
    return binary


def get_rapidocr_engine():
    global RAPIDOCR_ENGINE, RAPIDOCR_AVAILABLE
    if RAPIDOCR_AVAILABLE is False:
        return None
    if RAPIDOCR_ENGINE is not None:
        return RAPIDOCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR

        RAPIDOCR_ENGINE = RapidOCR()
        RAPIDOCR_AVAILABLE = True
        return RAPIDOCR_ENGINE
    except Exception:
        RAPIDOCR_AVAILABLE = False
        RAPIDOCR_ENGINE = None
        return None


def parse_temperature_candidate(text):
    if not text:
        return None
    normalized = str(text).replace(" ", "").replace(",", ".")
    matches = re.findall(r"(-?\d{2,3}\.\d)", normalized)
    if not matches:
        matches = re.findall(r"(-?\d{2,3})", normalized)
    for raw in matches:
        try:
            value = float(raw)
        except Exception:
            continue
        if value > 150.0 and value <= 999.0:
            value = round(value / 10.0, 1)
        if 10.0 <= value <= 180.0:
            return {
                "text": raw,
                "temperatureC": round(value, 1),
            }
    return None


def rotate_point_ccw(x, y, width, height):
    return float(y), float(width - 1 - x)


def extract_thermal_temperature_rapidocr(frame, focus_point):
    engine = get_rapidocr_engine()
    if engine is None or frame is None:
        return None

    height, width = frame.shape[:2]
    crops = [
        ("top-right", [int(width * 0.45), 0, width, int(height * 0.24)], False),
        ("top-band", [int(width * 0.30), 0, width, int(height * 0.28)], False),
        ("top-full", [0, 0, width, int(height * 0.24)], False),
    ]
    if focus_point is not None:
        cx = int(round(float(focus_point[0])))
        cy = int(round(float(focus_point[1])))
        crops.extend(
            [
                ("focus1", [cx - 92, cy - 124, cx + 72, cy + 78], True),
                ("focus2", [cx - 82, cy - 102, cx + 58, cy + 68], True),
            ]
        )
    best = None
    for crop_name, rect, rotate_first in crops:
        crop, crop_rect = crop_frame_rect(frame, rect)
        if crop is None or crop.size == 0:
            continue
        working = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotate_first else crop
        variants = []
        for scale in (3, 4, 5, 6):
            up = cv2.resize(working, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
            _, binary = cv2.threshold(blackhat, 16, 255, cv2.THRESH_BINARY)
            variants.append((f"{crop_name}-raw-x{scale}", up, scale))
            variants.append((f"{crop_name}-th-x{scale}", binary, scale))

        for variant_name, image, scale in variants:
            try:
                results, _ = engine(image)
            except Exception:
                continue
            if not results:
                continue
            for item in results:
                if len(item) < 3:
                    continue
                points, text, score = item[0], str(item[1]), float(item[2])
                parsed = parse_temperature_candidate(text)
                if parsed is None:
                    continue
                bbox_x = [float(pt[0]) for pt in points]
                bbox_y = [float(pt[1]) for pt in points]
                bbox_rot = [
                    min(bbox_x) / scale,
                    min(bbox_y) / scale,
                    max(bbox_x) / scale,
                    max(bbox_y) / scale,
                ]
                rx1, ry1, rx2, ry2 = bbox_rot
                if rotate_first:
                    p1 = rotate_point_ccw(rx1, ry2, crop.shape[1], crop.shape[0])
                    p2 = rotate_point_ccw(rx2, ry1, crop.shape[1], crop.shape[0])
                    abs_rect = [
                        crop_rect[0] + min(p1[0], p2[0]),
                        crop_rect[1] + min(p1[1], p2[1]),
                        crop_rect[0] + max(p1[0], p2[0]),
                        crop_rect[1] + max(p1[1], p2[1]),
                    ]
                else:
                    abs_rect = [
                        crop_rect[0] + rx1,
                        crop_rect[1] + ry1,
                        crop_rect[0] + rx2,
                        crop_rect[1] + ry2,
                    ]
                confidence = float(score)
                if "th-" in variant_name:
                    confidence += 0.18
                if crop_name.startswith("top-"):
                    confidence += 0.22
                if parsed["temperatureC"] >= 20.0:
                    confidence += 0.08
                candidate = {
                    "ok": True,
                    "text": text,
                    "temperatureC": parsed["temperatureC"],
                    "confidence": round(confidence, 4),
                    "region": f"rapidocr:{variant_name}",
                    "bbox": [int(round(v)) for v in abs_rect],
                }
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
    return best


def classify_thermal_character(mask, box, line_shape):
    line_h, line_w = line_shape[:2]
    x, y, w, h = box
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return "", 0.0
    if h <= max(5, int(line_h * 0.32)) and w <= max(7, int(line_w * 0.10)) and y >= int(line_h * 0.45):
        return ".", 0.92
    if h <= max(5, int(line_h * 0.26)) and w >= max(5, int(h * 1.2)):
        return "-", 0.84

    normalized = normalize_binary_mask(mask)
    best_char = ""
    best_score = -1.0
    for char, variants in thermal_char_templates().items():
        for variant in variants:
            union = np.logical_or(normalized > 0, variant > 0).sum()
            if union <= 0:
                continue
            intersection = np.logical_and(normalized > 0, variant > 0).sum()
            score = float(intersection / union)
            fill_delta = abs(
                float(np.count_nonzero(normalized)) / max(normalized.size, 1)
                - float(np.count_nonzero(variant)) / max(variant.size, 1)
            )
            score -= fill_delta * 0.16
            if score > best_score:
                best_score = score
                best_char = char
    return best_char, max(0.0, best_score)


def temperature_text_mask(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hsv, np.array([0, 0, 150], dtype=np.uint8), np.array([179, 120, 255], dtype=np.uint8))
    yellow = cv2.inRange(hsv, np.array([10, 25, 130], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
    red_a = cv2.inRange(hsv, np.array([0, 65, 110], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    red_b = cv2.inRange(hsv, np.array([165, 65, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    gray_mask = cv2.threshold(gray, 188, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.bitwise_or(mask, red_a)
    mask = cv2.bitwise_or(mask, red_b)
    mask = cv2.bitwise_or(mask, gray_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8), iterations=1)
    return ensure_binary_foreground(mask)


def segment_temperature_characters(binary):
    if binary is None or binary.size == 0:
        return []
    line_h, line_w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 2:
            continue
        if w >= line_w * 0.90 and h >= line_h * 0.90:
            continue
        if h < max(3, int(line_h * 0.18)):
            if not (w <= max(7, int(line_w * 0.10)) and y >= int(line_h * 0.45)):
                continue
        if w < 2 or h < 2:
            continue
        rects.append((x, y, w, h))
    rects.sort(key=lambda item: item[0])

    expanded = []
    for x, y, w, h in rects:
        if w <= max(14, int(h * 1.18)):
            expanded.append((x, y, w, h))
            continue
        blob = binary[y : y + h, x : x + w]
        projection = np.count_nonzero(blob, axis=0).astype(np.float32)
        if projection.size < 4:
            expanded.append((x, y, w, h))
            continue
        inner = projection[1:-1]
        split_index = None
        low_columns = np.where(inner <= max(1.0, h * 0.10))[0]
        if low_columns.size:
            target = float(w - 1) * 0.5
            split_index = int(min(low_columns + 1, key=lambda item: abs(float(item) - target)))
        else:
            target_start = max(1, int(round(w * 0.28)))
            target_end = min(w - 1, int(round(w * 0.72)))
            if target_end > target_start:
                band = projection[target_start:target_end]
                split_index = int(target_start + int(np.argmin(band)))
        if split_index is None or split_index <= 1 or split_index >= w - 1:
            expanded.append((x, y, w, h))
            continue
        left = blob[:, :split_index]
        right = blob[:, split_index:]
        if int(np.count_nonzero(left)) == 0 or int(np.count_nonzero(right)) == 0:
            expanded.append((x, y, w, h))
            continue
        expanded.append((x, y, split_index, h))
        expanded.append((x + split_index, y, w - split_index, h))
    expanded.sort(key=lambda item: item[0])
    return expanded


def decode_temperature_line(line_patch):
    if line_patch is None or line_patch.size == 0:
        return None
    gray = cv2.cvtColor(line_patch, cv2.COLOR_BGR2GRAY)
    mask = temperature_text_mask(line_patch)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    variants = [mask, otsu, 255 - otsu, adaptive, 255 - adaptive]
    best = None
    for variant in variants:
        binary = ensure_binary_foreground(variant)
        rects = segment_temperature_characters(binary)
        if not rects:
            continue
        chars = []
        scores = []
        for rect in rects:
            x, y, w, h = rect
            char_mask = binary[y : y + h, x : x + w]
            char, score = classify_thermal_character(char_mask, rect, binary.shape)
            if not char:
                continue
            chars.append(char)
            scores.append(score)
        if not chars:
            continue
        text = "".join(chars)
        text = text.replace("..", ".").replace(",.", ".").replace("C.", "C")
        match = THERMAL_TEMPERATURE_RE.search(text)
        if match is None:
            digits_only = re.sub(r"[^0-9-]", "", text)
            if len(digits_only) == 3 and digits_only.isdigit():
                digits_value = int(digits_only)
                if 150 <= digits_value <= 999:
                    text = f"{digits_value / 10.0:.1f}"
                    match = THERMAL_TEMPERATURE_RE.search(text)
        if match is None:
            continue
        raw_value = match.group(1).replace(",", ".")
        try:
            value = float(raw_value)
        except Exception:
            continue
        if value > 150.0 and value <= 999.0:
            value = round(value / 10.0, 1)
        if value < -40.0 or value > 300.0:
            continue
        confidence = float(sum(scores) / max(len(scores), 1))
        quality = confidence + min(len(chars), 5) * 0.035
        if 10.0 <= value <= 180.0:
            quality += 0.08
        candidate = {
            "text": text,
            "temperatureC": round(value, 1),
            "confidence": round(confidence, 4),
            "charCount": len(chars),
            "quality": round(quality, 4),
        }
        if best is None or candidate["quality"] > best["quality"]:
            best = candidate
    return best


def build_temperature_search_regions(frame_shape, focus_point=None):
    height, width = frame_shape[:2]
    top_h = max(48, int(round(height * 0.26)))
    bottom_h = max(44, int(round(height * 0.20)))
    regions = [
        ("top-full", [0, 0, width, min(height, top_h)]),
        ("top-left", [0, 0, min(width, int(round(width * 0.60))), min(height, int(round(height * 0.30)))]),
        ("top-right", [max(0, int(round(width * 0.40))), 0, width, min(height, int(round(height * 0.30)))]),
        ("bottom-full", [0, max(0, height - bottom_h), width, height]),
        ("bottom-right", [max(0, int(round(width * 0.40))), max(0, height - bottom_h), width, height]),
    ]
    if focus_point is not None:
        cx = int(round(float(focus_point[0])))
        cy = int(round(float(focus_point[1])))
        regions.append(
            (
                "focus-band",
                [
                    max(0, cx - int(round(width * 0.28))),
                    max(0, cy - int(round(height * 0.30))),
                    min(width, cx + int(round(width * 0.28))),
                    min(height, cy + int(round(height * 0.08))),
                ],
            )
        )
        regions.append(
            (
                "crosshair-vertical-right",
                [
                    min(width - 1, max(0, cx + 2)),
                    max(0, cy - int(round(height * 0.16))),
                    min(width, cx + int(round(width * 0.18))),
                    min(height, cy + int(round(height * 0.12))),
                ],
            )
        )
        regions.append(
            (
                "crosshair-vertical-left",
                [
                    max(0, cx - int(round(width * 0.18))),
                    max(0, cy - int(round(height * 0.16))),
                    max(1, min(width, cx - 2)),
                    min(height, cy + int(round(height * 0.12))),
                ],
            )
        )
    return regions


def extract_thermal_temperature_ocr(frame, focus_point=None):
    if frame is None:
        return {"ok": False, "error": "thermal frame unavailable"}
    rapidocr_result = extract_thermal_temperature_rapidocr(frame, focus_point)
    if rapidocr_result is not None:
        return rapidocr_result
    best = None
    for region_name, rect in build_temperature_search_regions(frame.shape, focus_point=focus_point):
        patch, patch_rect = crop_frame_rect(frame, rect)
        if patch is None or patch.size == 0:
            continue
        orientations = [("native", patch, patch_rect, False)]
        if "vertical" in region_name:
            orientations.insert(0, ("rot90ccw", cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE), patch_rect, True))
            orientations.insert(1, ("rot90cw", cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE), patch_rect, True))

        for orientation_name, oriented_patch, oriented_rect, is_rotated in orientations:
            region_mask = temperature_text_mask(oriented_patch)
            merged = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if area < 18 or w < 12 or h < 8:
                    continue
                if w > oriented_patch.shape[1] * 0.98 and h > oriented_patch.shape[0] * 0.86:
                    continue
                bias = 0.0
                if "top" in region_name:
                    bias += 0.06
                if y <= oriented_patch.shape[0] * 0.22:
                    bias += 0.04
                if "vertical" in region_name:
                    bias += 0.28
                line_candidates.append({"rect": [x, y, x + w, y + h], "bias": bias})
            if not line_candidates:
                line_candidates = [{"rect": [0, 0, oriented_patch.shape[1], oriented_patch.shape[0]], "bias": 0.0}]

            for item in line_candidates:
                line_rect = expand_rect(item["rect"], 4, 4, oriented_patch.shape)
                if line_rect is None:
                    continue
                line_patch, line_rect = crop_frame_rect(oriented_patch, line_rect)
                parsed = decode_temperature_line(line_patch)
                if parsed is None:
                    continue
                if is_rotated:
                    abs_rect = [patch_rect[0], patch_rect[1], patch_rect[2], patch_rect[3]]
                else:
                    abs_rect = [
                        patch_rect[0] + line_rect[0],
                        patch_rect[1] + line_rect[1],
                        patch_rect[0] + line_rect[2],
                        patch_rect[1] + line_rect[3],
                    ]
                confidence = float(parsed["confidence"]) + float(item["bias"])
                candidate = {
                    "ok": True,
                    "text": parsed["text"],
                    "temperatureC": parsed["temperatureC"],
                    "confidence": round(confidence, 4),
                    "region": region_name if not is_rotated else f"{region_name}:{orientation_name}",
                    "bbox": [int(v) for v in abs_rect],
                }
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
    if best is not None:
        return best
    return {"ok": False, "error": "thermal temperature text not recognized"}


def build_report_fusion_focus_image(frame, analysis):
    if frame is None or analysis is None:
        return None
    nearest = analysis.get("nearestComponent") or {}
    component_rect = None
    image_bounds = nearest.get("imageBounds")
    if image_bounds and len(image_bounds) == 4:
        x1, y1, x2, y2 = [float(v) for v in image_bounds]
        component_rect = [x1, y1, x2, y2]
    board_rect = quad_to_rect(analysis.get("visibleQuad"))
    crop, crop_rect = crop_board_region(frame, board_rect, pad_ratio=0.05)
    if (crop is None or crop_rect is None) and board_rect is None:
        focus_rect = union_rects([component_rect])
        center = analysis.get("imagePoint") or nearest.get("imagePoint")
        crop, crop_rect = crop_focus_region(
            frame,
            center=center,
            focus_rect=focus_rect,
            min_size=(820, 520),
            aspect_ratio=1.58,
            pad_ratio=0.08,
        )
    if crop is None or crop_rect is None:
        return None
    x1, y1, _, _ = crop_rect
    if component_rect is not None:
        bx1 = int(round(float(component_rect[0]) - x1))
        by1 = int(round(float(component_rect[1]) - y1))
        bx2 = int(round(float(component_rect[2]) - x1))
        by2 = int(round(float(component_rect[3]) - y1))
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (255, 220, 0), 2, cv2.LINE_AA)
    return crop


def build_report_thermal_focus_image(frame, analysis):
    if frame is None or analysis is None:
        return None
    ocr = analysis.get("thermalOcr") or {}
    hotspot_rect = xywh_to_xyxy(analysis.get("bboxThermal"))
    board_rect = quad_to_rect(analysis.get("thermalQuad"))
    crop, crop_rect = crop_board_region(frame, board_rect, pad_ratio=0.08)
    if (crop is None or crop_rect is None) and board_rect is None:
        focus_rect = union_rects([hotspot_rect, ocr.get("bbox")])
        center = analysis.get("thermalPoint") or analysis.get("crosshairPoint")
        crop, crop_rect = crop_focus_region(
            frame,
            center=center,
            focus_rect=focus_rect,
            min_size=(180, 160),
            aspect_ratio=1.45,
            pad_ratio=0.22,
        )
    if crop is None or crop_rect is None:
        return None
    x1, y1, _, _ = crop_rect
    if hotspot_rect is not None:
        bx1 = int(round(float(hotspot_rect[0]) - x1))
        by1 = int(round(float(hotspot_rect[1]) - y1))
        bx2 = int(round(float(hotspot_rect[2]) - x1))
        by2 = int(round(float(hotspot_rect[3]) - y1))
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 255), 2, cv2.LINE_AA)
    if ocr.get("bbox") is not None:
        ox1 = int(round(float(ocr["bbox"][0]) - x1))
        oy1 = int(round(float(ocr["bbox"][1]) - y1))
        ox2 = int(round(float(ocr["bbox"][2]) - x1))
        oy2 = int(round(float(ocr["bbox"][3]) - y1))
        cv2.rectangle(crop, (ox1, oy1), (ox2, oy2), (255, 255, 255), 2, cv2.LINE_AA)
        text = f"OCR {ocr.get('temperatureC', '--')}C"
        text_y = max(14, crop.shape[0] - 12)
        draw_text_with_backing(crop, text, np.array([8, text_y], dtype=np.float32), (255, 255, 255), 0.32, 1)
    return crop


def build_report_image_assets(analysis, visible_frame=None, thermal_frame=None, stem="diagnostic-report"):
    assets = {}
    fusion_focus = build_report_fusion_focus_image(visible_frame, analysis)
    if fusion_focus is not None:
        fusion_path = save_report_image(fusion_focus, f"{stem}-fusion-focus")
        if fusion_path is not None:
            assets["fusionFocus"] = {
                "path": str(fusion_path),
                "caption": "融合定位关键区域",
            }
    thermal_focus = build_report_thermal_focus_image(thermal_frame, analysis)
    if thermal_focus is not None:
        thermal_path = save_report_image(thermal_focus, f"{stem}-thermal-focus")
        if thermal_path is not None:
            assets["thermalFocus"] = {
                "path": str(thermal_path),
                "caption": "热像热点与温度文字",
            }
    return assets


def summarize_yolo_obb_state(payload):
    summary = dict(payload or {})
    summary["detections"] = []
    return summary


def iter_detection_tiles(image, tile_grid=(2, 2), overlap_ratio=0.18):
    height, width = image.shape[:2]
    yield image, (0, 0)
    rows, cols = tile_grid
    if rows <= 1 and cols <= 1:
        return
    tile_w = max(int(np.ceil(width / cols)), 64)
    tile_h = max(int(np.ceil(height / rows)), 64)
    overlap_w = int(tile_w * overlap_ratio)
    overlap_h = int(tile_h * overlap_ratio)
    x_starts = []
    y_starts = []
    for col in range(cols):
        start_x = min(max(col * tile_w - overlap_w, 0), max(width - tile_w, 0))
        x_starts.append(start_x)
    for row in range(rows):
        start_y = min(max(row * tile_h - overlap_h, 0), max(height - tile_h, 0))
        y_starts.append(start_y)
    for start_y in sorted(set(y_starts)):
        for start_x in sorted(set(x_starts)):
            tile = image[start_y : min(start_y + tile_h, height), start_x : min(start_x + tile_w, width)]
            if tile.size == 0:
                continue
            yield tile, (start_x, start_y)


def run_rknn_obb_detection(image, model_path, conf_threshold=0.22, nms_threshold=0.10):
    try:
        ctypes.CDLL("/usr/lib/librknnrt.so", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    try:
        from rknnlite.api import RKNNLite as RKNN
    except Exception as exc:
        raise RuntimeError(f"rknnlite import failed: {exc}")

    rknn = RKNN()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    nms_boxes = []
    confidences = []
    labels = []
    try:
        for tile, origin in iter_detection_tiles(image, tile_grid=(2, 2), overlap_ratio=0.18):
            img_canvas, scale, dx, dy = letterbox_resize(tile)
            input_data = np.expand_dims(cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB), 0)
            outputs = rknn.inference(inputs=[input_data])
            if not outputs:
                continue
            data = outputs[0][0]
            cls_scores = data[4:20, :]
            probs = 1 / (1 + np.exp(-cls_scores))
            max_probs = np.max(probs, axis=0)
            cls_ids = np.argmax(probs, axis=0)
            top_indices = np.argsort(max_probs)[-240:]

            for idx in top_indices:
                score = float(max_probs[idx])
                if score <= conf_threshold:
                    continue
                cx = float((data[0, idx] - dx) / scale) + origin[0]
                cy = float((data[1, idx] - dy) / scale) + origin[1]
                w = float(data[2, idx] / scale)
                h = float(data[3, idx] / scale)
                angle = float(data[20, idx]) * 180.0 / np.pi
                if min(w, h) <= 2:
                    continue
                if max(w, h) / (min(w, h) + 1e-6) > 12:
                    continue
                nms_boxes.append(((cx, cy), (w, h), angle))
                confidences.append(score)
                labels.append(f"cls_{int(cls_ids[idx])}")

        detections = []
        if nms_boxes:
            indices = cv2.dnn.NMSBoxesRotated(nms_boxes, confidences, conf_threshold, nms_threshold)
            if len(indices) > 0:
                for item in indices:
                    idx = item[0] if isinstance(item, (list, tuple, np.ndarray)) else item
                    rect = nms_boxes[idx]
                    box_points = cv2.boxPoints(rect).tolist()
                    detections.append(
                        {
                            "label": labels[idx],
                            "score": float(confidences[idx]),
                            "quad": [[float(p[0]), float(p[1])] for p in box_points],
                        }
                    )
        return detections
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def find_yolo_obb_model():
    preferred = [
        Path("/home/elf/Yolo_Models/pcb_obb_fp16.rknn"),
        Path("/home/elf/pcb_fp16.rknn"),
        Path("/home/elf/pcbfp16_2.rknn"),
        Path("/home/elf/pcb2.rknn"),
        Path("/home/elf/pcb.rknn"),
    ]
    for match in preferred:
        if match.is_file():
            return match

    search_roots = [ROOT, ROOT.parent, Path("/home/elf")]
    patterns = ["*obb*.rknn", "*yolo*.rknn", "*obb*.pt", "*yolo*.pt", "*obb*.onnx"]
    for base in search_roots:
        if not base.exists():
            continue
        for pattern in patterns:
            for match in sorted(base.rglob(pattern)):
                if match.is_file():
                    return match
    return None


def run_yolo_obb_detection(image_path, result_path):
    model_path = find_yolo_obb_model()
    if model_path is None:
        raise RuntimeError("no YOLO OBB model found on board")

    code = f"""
import json
from pathlib import Path

image_path = Path({json.dumps(str(image_path))})
result_path = Path({json.dumps(str(result_path))})
model_path = Path({json.dumps(str(model_path))})

payload = {{
    "enabled": True,
    "running": False,
    "available": False,
    "message": "not-run",
    "savedAt": __import__("time").time(),
    "modelPath": str(model_path),
    "count": 0,
    "detections": [],
}}
"""
    if model_path.suffix.lower() == ".rknn":
        code += """
import ctypes
import cv2
import numpy as np

try:
    ctypes.CDLL('/usr/lib/librknnrt.so', mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

try:
    from rknnlite.api import RKNNLite as RKNN
except Exception as exc:
    raise RuntimeError(f'rknnlite import failed: {exc}')

def letterbox_resize(image, size=(640, 640)):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = image_resized
    return canvas, scale, dx, dy

rknn = RKNN()
ret = rknn.load_rknn(str(model_path))
if ret != 0:
    raise RuntimeError(f'load_rknn failed: {ret}')
ret = rknn.init_runtime()
if ret != 0:
    raise RuntimeError(f'init_runtime failed: {ret}')

frame = cv2.imread(str(image_path))
if frame is None:
    raise RuntimeError('failed to read input image')

img_canvas, scale, dx, dy = letterbox_resize(frame)
input_data = np.expand_dims(cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB), 0)
outputs = rknn.inference(inputs=[input_data])
if not outputs:
    raise RuntimeError('empty RKNN outputs')

data = outputs[0][0]
cls_scores = data[4:20, :]
probs = 1 / (1 + np.exp(-cls_scores))
max_probs = np.max(probs, axis=0)
cls_ids = np.argmax(probs, axis=0)
top_indices = np.argsort(max_probs)[-120:]
nms_boxes, confidences, labels = [], [], []

for i in top_indices:
    score = float(max_probs[i])
    if score <= 0.35:
        continue
    cx = float((data[0, i] - dx) / scale)
    cy = float((data[1, i] - dy) / scale)
    w = float(data[2, i] / scale)
    h = float(data[3, i] / scale)
    angle = float(data[20, i]) * 180.0 / np.pi
    if min(w, h) <= 1:
        continue
    if max(w, h) / (min(w, h) + 1e-6) > 8:
        continue
    nms_boxes.append(((cx, cy), (w, h), angle))
    confidences.append(score)
    labels.append(f'cls_{int(cls_ids[i])}')

if nms_boxes:
    indices = cv2.dnn.NMSBoxesRotated(nms_boxes, confidences, 0.35, 0.10)
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            rect = nms_boxes[idx]
            box_points = cv2.boxPoints(rect).tolist()
            payload["detections"].append({
                "label": labels[idx],
                "score": float(confidences[idx]),
                "quad": [[float(p[0]), float(p[1])] for p in box_points],
            })

payload["available"] = True
payload["message"] = "ok"
payload["count"] = len(payload["detections"])
result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
rknn.release()
"""
    else:
        code += """
try:
    from ultralytics import YOLO
except Exception as exc:
    raise RuntimeError(f'ultralytics import failed: {exc}')

model = YOLO(str(model_path))
results = model.predict(source=str(image_path), verbose=False, save=False, imgsz=1280)
for result in results:
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        continue
    names = getattr(result, "names", {})
    points_list = obb.xyxyxyxy.cpu().numpy().tolist()
    cls_list = obb.cls.cpu().numpy().tolist() if obb.cls is not None else [None] * len(points_list)
    conf_list = obb.conf.cpu().numpy().tolist() if obb.conf is not None else [None] * len(points_list)
    for pts, cls_id, conf in zip(points_list, cls_list, conf_list):
        quad = [[float(p[0]), float(p[1])] for p in pts]
        label = names.get(int(cls_id), str(int(cls_id))) if cls_id is not None else "obb"
        payload["detections"].append({
            "label": label,
            "score": None if conf is None else float(conf),
            "quad": quad,
        })

payload["available"] = True
payload["message"] = "ok"
payload["count"] = len(payload["detections"])
result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
"""
    return subprocess.run(
        ["python3", "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=40,
    ), model_path


def draw_cross(frame, point, color):
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.line(frame, (x - 18, y), (x + 18, y), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x, y - 18), (x, y + 18), color, 3, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 6, color, 2, cv2.LINE_AA)


def extract_json_object_from_text(text):
    value = str(text or "").strip()
    if not value:
        return None
    candidates = re.findall(r"\{[\s\S]*\}", value)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def build_deepseek_prompt(report_payload):
    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    component_context = report_payload.get("componentContext") or {}
    nets = nearest.get("nets") or []
    candidate_summary = nearest.get("candidateSummary") or []
    nets_text = "\n".join(
        f"- 网线: {item.get('net', '--')} / 引脚: {item.get('pin', '--')} / 名称: {item.get('name', '--')}"
        for item in nets[:8]
    ) or "- 暂无网表信息"
    candidate_text = "\n".join(
        f"- {item.get('ref', '--')} / sumHeat={item.get('sumHeat', '--')} / meanHeat={item.get('meanHeat', '--')} / maxHeat={item.get('maxHeat', '--')}"
        for item in candidate_summary[:5]
    ) or "- 暂无候选器件热量排序"
    source_file = BOARD_MODEL.get("sourceFile") or "default-board-model"
    return (
        "你是 PCB 热诊断工程师。请基于以下结构化信息，输出详细、专业、全中文的诊断结果。\n"
        "要求:\n"
        "1. 只返回 JSON，不要输出 markdown。\n"
        "2. JSON 字段必须包含 summary、riskLevel、actions、detailedAnalysis、probabilities、componentProfile。\n"
        "3. summary 控制在 50 字以内，适合首页摘要。\n"
        "4. detailedAnalysis 使用 2 到 4 段中文分析。\n"
        "5. probabilities 是数组，每项包含 label 和 percent，总和尽量接近 100。\n"
        "6. componentProfile 包含 probableFunction、packageNotes、datasheetHints。\n"
        "7. actions 为中文字符串数组，给 3 条以内建议。\n"
        "8. 允许根据器件型号、封装、常见用途、网表关系和热像位置做合理推断，但不要伪造未确认的精确结论。\n"
        "9. 如果器件看起来像可检索型号，可以根据常见公开资料推断用途和参数方向，但不要伪造精确参数。\n\n"
        f"板卡编号: {report_payload.get('boardId', '--')}\n"
        f"坐标来源文件: {source_file}\n"
        f"生成时间: {report_payload.get('generatedAt', '--')}\n"
        f"热源定位来源: {analysis.get('hotspotSource', '--')}\n"
        f"热像准星点: {analysis.get('crosshairPoint', '--')}\n"
        f"热像热点点: {analysis.get('thermalPoint', '--')}\n"
        f"映射后的可见光点: {analysis.get('imagePoint', '--')}\n"
        f"PCB 坐标(mm): {analysis.get('boardPointMm', '--')}\n"
        f"OCR 温度文本: {thermal_ocr.get('text', '--')}\n"
        f"OCR 温度(℃): {thermal_ocr.get('temperatureC', '--')}\n"
        f"最近器件: {nearest.get('ref', '--')}\n"
        f"封装: {nearest.get('footprint', '--')}\n"
        f"型号: {nearest.get('device', '--')}\n"
        f"说明: {nearest.get('comment', '--')}\n"
        f"匹配模式: {nearest.get('matchMode', '--')}\n"
        f"热量分数: {nearest.get('heatScore', '--')}\n"
        f"热点框(热像): {analysis.get('bboxThermal', '--')}\n"
        f"器件图像边界(可见光): {nearest.get('imageBounds', '--')}\n"
        f"候选器件热量排名:\n{candidate_text}\n"
        "网表信息:\n"
        f"{nets_text}\n"
    )


def build_local_deepseek_prompt(report_payload):
    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    name_hint = str(nearest.get("name") or nearest.get("comment") or nearest.get("device") or "--").strip()
    comment_hint = str(nearest.get("comment") or nearest.get("name") or "--").strip()
    nets = nearest.get("nets") or []
    net_hint = " / ".join(
        f"{item.get('net', '--')}:{item.get('pin', '--')}"
        for item in nets[:3]
        if isinstance(item, dict)
    ) or "--"
    return (
        "你是PCB热诊断助手。只输出中文纯文本，不要JSON，不要markdown，不要解释推理。\n"
        "严格按4行回复，控制在180字内：\n"
        "概要: ...\n"
        "风险: 高/中/低\n"
        "概率: 原因A 60%; 原因B 30%; 原因C 10%\n"
        "建议: 1. ... 2. ...\n\n"
        "请优先根据器件型号、器件名称、器件说明、封装和网络连接，判断该器件的功能以及常见发热原因。\n"
        f"板卡: {report_payload.get('boardId', '--')}\n"
        f"热源: {analysis.get('hotspotSource', '--')}\n"
        f"准星: {analysis.get('crosshairPoint', '--')}\n"
        f"热点: {analysis.get('thermalPoint', '--')}\n"
        f"可见光点: {analysis.get('imagePoint', '--')}\n"
        f"温度: {thermal_ocr.get('temperatureC', '--')}C\n"
        f"位号: {nearest.get('ref', '--')}\n"
        f"型号: {nearest.get('device', '--')}\n"
        f"名称: {name_hint}\n"
        f"说明: {comment_hint}\n"
        f"封装: {nearest.get('footprint', '--')}\n"
        f"网络: {net_hint}\n"
        f"匹配: {nearest.get('matchMode', '--')}\n"
    )


def build_deepseek_prompt_v2(report_payload):
    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    component_context = report_payload.get("componentContext") or {}
    nets = nearest.get("nets") or []
    candidate_summary = nearest.get("candidateSummary") or []
    nets_text = "\n".join(
        f"- 网络: {item.get('net', '--')} / 引脚: {item.get('pin', '--')} / 名称: {item.get('name', '--')}"
        for item in nets[:8]
    ) or "- 暂无网表信息"
    candidate_text = "\n".join(
        f"- {item.get('ref', '--')} / sumHeat={item.get('sumHeat', '--')} / meanHeat={item.get('meanHeat', '--')} / maxHeat={item.get('maxHeat', '--')}"
        for item in candidate_summary[:5]
    ) or "- 暂无候选器件热量排名"
    source_file = BOARD_MODEL.get("sourceFile") or "default-board-model"
    return (
        "你是 PCB 热诊断工程师。请基于以下结构化信息，输出详细、专业、全中文的诊断结果。\n"
        "要求:\n"
        "1. 只返回 JSON，不要输出 markdown。\n"
        "2. JSON 字段必须包含 summary、riskLevel、actions、detailedAnalysis、probabilities、componentProfile。\n"
        "3. summary 控制在 50 字以内，适合首页摘要。\n"
        "4. detailedAnalysis 使用 2 到 4 段中文分析。\n"
        "5. probabilities 是数组，每项包含 label 和 percent，总和尽量接近 100。\n"
        "6. componentProfile 包含 probableFunction、packageNotes、datasheetHints。\n"
        "7. actions 为中文字符串数组，给 3 条以内建议。\n"
        "8. 允许根据器件型号、名称、说明、封装、网络关系、邻近器件和热像位置做合理推断，但不要伪造未经确认的精确结论。\n"
        "9. 如果器件型号看起来像可检索器件，可依据常见公开资料推断用途和参数方向，但不要伪造精确参数值。\n\n"
        f"板卡编号: {report_payload.get('boardId', '--')}\n"
        f"坐标来源文件: {source_file}\n"
        f"生成时间: {report_payload.get('generatedAt', '--')}\n"
        f"热源定位来源: {analysis.get('hotspotSource', '--')}\n"
        f"热像准星点: {analysis.get('crosshairPoint', '--')}\n"
        f"热像热点点: {analysis.get('thermalPoint', '--')}\n"
        f"映射后的可见光点: {analysis.get('imagePoint', '--')}\n"
        f"PCB 坐标(mm): {analysis.get('boardPointMm', '--')}\n"
        f"OCR 温度文本: {thermal_ocr.get('text', '--')}\n"
        f"OCR 温度(C): {thermal_ocr.get('temperatureC', '--')}\n"
        f"最近器件: {nearest.get('ref', '--')}\n"
        f"封装: {nearest.get('footprint', '--')}\n"
        f"型号: {nearest.get('device', '--')}\n"
        f"说明: {nearest.get('comment', '--')}\n"
        f"匹配模式: {nearest.get('matchMode', '--')}\n"
        f"热量分数: {nearest.get('heatScore', '--')}\n"
        f"热点框(热像): {analysis.get('bboxThermal', '--')}\n"
        f"器件图像边界(可见光): {nearest.get('imageBounds', '--')}\n"
        f"器件资料摘要: {component_context.get('summaryText', '--')}\n"
        f"器件名称: {component_context.get('name', nearest.get('name', '--'))}\n"
        f"器件标识: {component_context.get('identityText', '--')}\n"
        f"器件网络摘要: {component_context.get('netText', '--')}\n"
        f"器件邻近摘要: {component_context.get('nearbyText', '--')}\n"
        f"候选器件热量排名:\n{candidate_text}\n"
        "网表信息:\n"
        f"{nets_text}\n"
    )


def build_local_deepseek_prompt_v2(report_payload):
    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    component_context = report_payload.get("componentContext") or {}
    name_hint = str(component_context.get("name") or nearest.get("name") or nearest.get("comment") or nearest.get("device") or "--").strip()
    comment_hint = str(component_context.get("comment") or nearest.get("comment") or nearest.get("name") or "--").strip()
    identity_hint = str(component_context.get("identityText") or "--").strip()
    net_hint = str(component_context.get("netText") or "--").strip()
    nearby_hint = str(component_context.get("nearbyText") or "--").strip()
    return (
        "你是PCB热诊断助手。只输出中文纯文本，不要JSON，不要markdown，不要解释推理。\n"
        "严格按4行回复，控制在180字内：\n"
        "概要: ...\n"
        "风险: 高/中/低\n"
        "概率: 原因A 60%; 原因B 30%; 原因C 10%\n"
        "建议: 1. ... 2. ...\n\n"
        "请优先根据器件型号、器件名称、器件说明、封装、网络连接和邻近器件，判断该器件的功能以及常见发热原因。\n"
        f"板卡: {report_payload.get('boardId', '--')}\n"
        f"热源: {analysis.get('hotspotSource', '--')}\n"
        f"准星: {analysis.get('crosshairPoint', '--')}\n"
        f"热点: {analysis.get('thermalPoint', '--')}\n"
        f"可见光点: {analysis.get('imagePoint', '--')}\n"
        f"温度: {thermal_ocr.get('temperatureC', '--')}C\n"
        f"位号: {nearest.get('ref', '--')}\n"
        f"型号: {nearest.get('device', '--')}\n"
        f"名称: {name_hint}\n"
        f"说明: {comment_hint}\n"
        f"标识: {identity_hint}\n"
        f"封装: {nearest.get('footprint', '--')}\n"
        f"网络: {net_hint}\n"
        f"邻近器件: {nearby_hint}\n"
        f"匹配: {nearest.get('matchMode', '--')}\n"
    )


def parse_local_deepseek_text_payload(reply_text, report_payload):
    analysis = deepcopy((report_payload or {}).get("analysis") or {})
    nearest = deepcopy(analysis.get("nearestComponent") or {})
    clean_text = strip_ansi_text(reply_text).replace("\r", "\n").strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    summary_match = re.search(r"^\s*概要[:：]\s*(.+)$", clean_text, re.MULTILINE)
    risk_match = re.search(r"^\s*风险[:：]\s*([高中低])", clean_text, re.MULTILINE)
    probability_match = re.search(r"^\s*概率[:：]\s*(.+)$", clean_text, re.MULTILINE)
    actions_match = re.search(r"^\s*建议[:：]\s*(.+)$", clean_text, re.MULTILINE)

    summary = (summary_match.group(1).strip() if summary_match else "").strip(" .;；，,")
    ref = str(nearest.get("ref") or "--")
    temp_c = (analysis.get("thermalOcr") or {}).get("temperatureC")
    if not summary:
        if temp_c not in (None, ""):
            summary = f"热点位于 {ref}，约 {temp_c}C"
        else:
            summary = f"热点位于 {ref}"

    risk_map = {"高": "high", "中": "medium", "低": "low"}
    risk_level = risk_map.get((risk_match.group(1) if risk_match else "").strip(), "medium")

    probabilities = []
    probability_text = probability_match.group(1).strip() if probability_match else ""
    if probability_text:
        for label, percent in re.findall(r"([^;；]+?)\s+(\d{1,3})\s*%", probability_text):
            label_text = re.sub(r"^[\-\d\.\)\(、\s]+", "", label).strip(" ;；，,")
            try:
                percent_value = int(percent)
            except Exception:
                continue
            if label_text:
                probabilities.append({"label": label_text, "percent": max(0, min(percent_value, 100))})
    if not probabilities:
        probabilities = deepcopy(analysis.get("probabilities") or [])

    actions = []
    actions_text = actions_match.group(1).strip() if actions_match else ""
    if actions_text:
        parts = re.split(r"\s*(?:\d+\s*[.)]|[;；])\s*", actions_text)
        for part in parts:
            part = part.strip(" .;；，,\n\t")
            if part:
                actions.append(part)
    if not actions:
        actions = [
            f"复核 {ref} 周边供电与焊点",
            "对比空载和负载热像变化",
            "结合万用表继续排查上下游器件",
        ]

    has_structured_reply = bool(summary_match or risk_match or probability_match or actions_match)
    looks_meta = any(
        keyword in clean_text
        for keyword in (
            "用户发来",
            "他想让我",
            "首先，我需要",
            "接下来，我要考虑",
            "嗯，用户",
            "看起来他想",
        )
    )
    if (not clean_text) or looks_meta or (not has_structured_reply):
        clean_text = "\n".join(
            [
                f"热点来源: {analysis.get('hotspotSource', '--')}",
                f"器件: {ref}",
                f"型号: {nearest.get('device') or nearest.get('comment') or '--'}",
                f"封装: {nearest.get('footprint') or '--'}",
                f"OCR温度: {temp_c if temp_c not in (None, '') else '--'}C",
                f"定位方式: {nearest.get('matchMode') or '--'}",
                f"建议: {'；'.join(actions[:3])}",
            ]
        )

    return {
        "ok": True,
        "provider": "deepseek-local",
        "transport": "local-text",
        "providerLabel": "Local DeepSeek 1.5B",
        "summary": compact_report_text(summary, max_chars=120),
        "riskLevel": risk_level,
        "actions": actions[:3],
        "detailedAnalysis": compact_report_text(clean_text or summary, max_chars=600),
        "probabilities": probabilities[:4],
        "componentProfile": {
            "probableFunction": nearest.get("device") or nearest.get("comment") or "--",
            "packageNotes": nearest.get("footprint") or "--",
            "datasheetHints": nearest.get("name") or nearest.get("comment") or "--",
        },
    }


def call_remote_deepseek_summary(report_payload, config):
    payload = {
        "model": config.get("model") or DEEPSEEK_DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一名严谨的 PCB 故障诊断助手。"},
            {"role": "user", "content": build_deepseek_prompt_v2(report_payload)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    url = f"{config.get('baseUrl', DEEPSEEK_BASE_URL)}{DEEPSEEK_CHAT_PATH}"
    response = http_post_json_headers(
        url,
        payload,
        headers={"Authorization": f"Bearer {config['apiKey']}"},
        timeout=35,
    )
    message = (((response.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not message:
        return {"ok": False, "error": "deepseek returned empty content", "provider": "deepseek-api", "transport": "api", "providerLabel": "DeepSeek API", "raw": response}
    parsed = extract_json_object_from_text(message)
    if parsed is None:
        parsed = {
            "summary": str(message).strip(),
            "riskLevel": "中",
            "actions": [],
            "detailedAnalysis": str(message).strip(),
            "probabilities": [],
            "componentProfile": {},
        }
    parsed["ok"] = True
    parsed["provider"] = "deepseek-api"
    parsed["transport"] = "api"
    parsed["providerLabel"] = "DeepSeek API"
    parsed["model"] = payload["model"]
    return parsed


def call_local_rkllm_server_summary(report_payload, config):
    endpoint = str(config.get("serverUrl") or "").strip()
    if not endpoint:
        raise RuntimeError("local rkllm server url missing")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
    response = http_post_json_headers(
        endpoint,
        {
            "model": str(config.get("serverModel") or "rkllm"),
            "messages": [{"role": "user", "content": build_deepseek_prompt_v2(report_payload)}],
            "stream": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": min(int(config.get("maxNewTokens") or LOCAL_LLM_MAX_NEW_TOKENS), 1024),
        },
        headers={"Authorization": "not_required"},
        timeout=LOCAL_LLM_TIMEOUT,
    )
    content = ""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            content = str((((choices[0] or {}).get("message") or {}).get("content")) or "").strip()
    parsed = extract_json_object_from_text(content)
    if parsed is None:
        parsed = {
            "summary": compact_report_text(content or "local rkllm server returned no parseable JSON", max_chars=120),
            "riskLevel": "medium",
            "actions": [],
            "detailedAnalysis": compact_report_text(content or "local rkllm server returned no detailed output", max_chars=260),
            "probabilities": [],
            "componentProfile": {},
        }
    parsed["ok"] = True
    parsed["provider"] = "rkllm-server"
    parsed["transport"] = "local-server"
    parsed["providerLabel"] = "本地 RKLLM Server"
    parsed["model"] = str(config.get("serverModel") or "rkllm")
    parsed["rawOutputPreview"] = compact_report_text(content, max_chars=260)
    return parsed


def call_local_deepseek_summary(report_payload):
    config = local_llm_config()
    if not config.get("enabled"):
        return {
            "ok": False,
            "error": "local llm runtime not found",
            "provider": "deepseek-local",
            "transport": "local",
            "providerLabel": "本地 DeepSeek 1.5B",
        }
    try:
        server_error = None
        server_url = str(config.get("serverUrl") or "").strip()
        if server_url and network_endpoint_reachable(server_url, timeout=2.0):
            try:
                return call_local_rkllm_server_summary(report_payload, config)
            except Exception as exc:
                server_error = str(exc)
        elif server_url:
            server_error = "local rkllm server unavailable"
        env = os.environ.copy()
        lib_path = str(LOCAL_LLM_ROOT / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        result = subprocess.run(
            [
                str(LOCAL_LLM_BINARY),
                str(LOCAL_LLM_MODEL),
                str(config["maxNewTokens"]),
                str(config["maxContextLen"]),
            ],
            input=(build_deepseek_prompt_v2(report_payload) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(LOCAL_LLM_ROOT),
            env=env,
            timeout=LOCAL_LLM_TIMEOUT,
        )
        combined = (result.stdout or b"").decode("utf-8", "ignore")
        if result.stderr:
            combined = combined + ("\n" if combined else "") + result.stderr.decode("utf-8", "ignore")
        parsed = extract_json_object_from_text(combined)
        if result.returncode not in (0, None) and parsed is None:
            error_text = f"local llm exited with code {result.returncode}"
            if int(result.returncode) == -9:
                error_text = "local llm exited with code -9 (likely OOM, reduce localMaxNewTokens/localMaxContextLen or enable swap)"
            return {
                "ok": False,
                "error": error_text + (f" | server: {server_error}" if server_error else ""),
                "provider": "deepseek-local",
                "transport": "local",
                "providerLabel": "本地 DeepSeek 1.5B",
                "model": LOCAL_LLM_MODEL.name,
                "rawOutputPreview": compact_report_text(combined, max_chars=260),
            }
        if parsed is None:
            parsed = {
                "summary": compact_report_text(combined or "本地模型未返回可解析 JSON。", max_chars=120),
                "riskLevel": "中",
                "actions": [],
                "detailedAnalysis": compact_report_text(combined or "本地模型未返回可解析内容。", max_chars=260),
                "probabilities": [],
                "componentProfile": {},
            }
        parsed["ok"] = True
        parsed["provider"] = "deepseek-local"
        parsed["transport"] = "local"
        parsed["providerLabel"] = "本地 DeepSeek 1.5B"
        parsed["model"] = LOCAL_LLM_MODEL.name
        parsed["rawOutputPreview"] = compact_report_text(combined, max_chars=260)
        return parsed
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "provider": "deepseek-local",
            "transport": "local",
            "providerLabel": "本地 DeepSeek 1.5B",
            "model": LOCAL_LLM_MODEL.name,
        }


def build_fallback_llm_summary(report_payload, base_result=None, provider="rules-fallback", provider_label="规则回退"):
    analysis = deepcopy((report_payload or {}).get("analysis") or {})
    nearest = deepcopy(analysis.get("nearestComponent") or {})
    thermal_ocr = deepcopy(analysis.get("thermalOcr") or {})
    temperature = thermal_ocr.get("temperatureC")
    ref = str(nearest.get("ref") or "--")
    device = str(nearest.get("device") or nearest.get("name") or nearest.get("comment") or "--")
    summary_parts = [f"热点位于 {ref}"]
    if temperature not in (None, ""):
        summary_parts.append(f"约 {temperature}°C")
    if nearest.get("matchMode"):
        summary_parts.append(f"匹配方式 {nearest.get('matchMode')}")
    detailed_lines = [
        f"热点来源: {analysis.get('hotspotSource', '--')}",
        f"PCB 坐标(mm): {analysis.get('boardPointMm', '--')}",
        f"器件: {ref}",
        f"型号/说明: {device}",
        f"封装: {nearest.get('footprint', '--')}",
        f"层面: {nearest.get('layer', '--')}",
        f"OCR 温度: {thermal_ocr.get('text', '--')}",
    ]
    if base_result and base_result.get("error"):
        detailed_lines.append(f"模型调用情况: {base_result.get('error')}")
    actions = [
        f"优先复核 {ref} 及其周边供电与焊点",
        "对比空载与负载热像，确认热点是否稳定复现",
        "结合示波器或万用表继续排查同网络上下游器件",
    ]
    return {
        "ok": True,
        "provider": provider,
        "transport": "fallback",
        "providerLabel": provider_label,
        "summary": "，".join(summary_parts),
        "riskLevel": "medium" if nearest else "low",
        "actions": actions,
        "detailedAnalysis": "\n".join(detailed_lines),
        "probabilities": deepcopy(analysis.get("probabilities") or []),
        "componentProfile": {
            "probableFunction": nearest.get("device") or nearest.get("comment") or "--",
            "packageNotes": nearest.get("footprint") or "--",
            "datasheetHints": nearest.get("name") or nearest.get("comment") or "--",
        },
        "rawOutputPreview": (base_result or {}).get("rawOutputPreview"),
        "fallbackReason": None if not base_result else base_result.get("error"),
        "checkedRoots": None if not base_result else base_result.get("checkedRoots"),
        "model": None if not base_result else base_result.get("model"),
    }


def strip_ansi_text(text):
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", str(text or ""))


def extract_local_llm_reply_text(transcript, marker_index=0):
    text = strip_ansi_text(transcript)[int(marker_index or 0) :]
    robot_match = re.search(r"robot:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not robot_match:
        return ""
    reply = robot_match.group(1)
    reply = re.split(r"\nuser:\s*", reply, maxsplit=1, flags=re.IGNORECASE)[0]
    reply = reply.replace("<think>", "").replace("</think>", "")
    return reply.strip()


def run_local_llm_text_session(prompt_text, local_root, local_binary, local_model, config, env):
    import pty
    import select

    master_fd, slave_fd = pty.openpty()
    process = None
    transcript = ""
    try:
        process = subprocess.Popen(
            [
                str(local_binary),
                str(local_model),
                str(config["maxNewTokens"]),
                str(config["maxContextLen"]),
            ],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(local_root),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        try:
            os.close(slave_fd)
        except Exception:
            pass

    prompt_sent = False
    prompt_marker_index = 0
    answer_started = False
    last_data_at = time.time()
    started_at = time.time()

    try:
        while time.time() - started_at < LOCAL_LLM_TIMEOUT:
            ready, _, _ = select.select([master_fd], [], [], 0.4)
            if master_fd in ready:
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                transcript += chunk.decode("utf-8", "ignore")
                last_data_at = time.time()

            clean_text = strip_ansi_text(transcript)
            if not prompt_sent:
                if (
                    "user:" in clean_text.lower()
                    or "rkllm init success" in clean_text.lower()
                    or "可输入以下问题" in clean_text
                ):
                    if ("user:" in clean_text.lower()) or (time.time() - last_data_at > 0.8):
                        prompt_marker_index = len(clean_text)
                        os.write(master_fd, prompt_text.encode("utf-8", "ignore") + b"\n")
                        prompt_sent = True
                        last_data_at = time.time()
                continue

            segment = strip_ansi_text(transcript)[prompt_marker_index:]
            if "robot:" in segment.lower():
                answer_started = True
            if answer_started and re.search(r"\nuser:\s*$", segment, re.IGNORECASE):
                break
            if answer_started and (time.time() - last_data_at) > 2.8:
                break

        try:
            os.write(master_fd, b"\x03")
        except Exception:
            pass
        time.sleep(0.4)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()
    finally:
        try:
            if process is not None:
                process.wait(timeout=3)
        except Exception:
            try:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        try:
            os.close(master_fd)
        except Exception:
            pass

    return {
        "transcript": transcript,
        "reply": extract_local_llm_reply_text(transcript, prompt_marker_index),
        "returncode": None if process is None else process.poll(),
    }


def call_local_deepseek_summary(report_payload):
    config = local_llm_config()
    if not config.get("enabled"):
        return {
            "ok": False,
            "error": "local llm runtime not found",
            "provider": "deepseek-local",
            "transport": "local",
            "providerLabel": "本地 DeepSeek 1.5B",
            "checkedRoots": config.get("checkedRoots") or [],
        }
    try:
        server_error = None
        server_url = str(config.get("serverUrl") or "").strip()
        if server_url and network_endpoint_reachable(server_url, timeout=2.0):
            try:
                return call_local_rkllm_server_summary(report_payload, config)
            except Exception as exc:
                server_error = str(exc)
        elif server_url:
            server_error = "local rkllm server unavailable"

        local_root = Path(config.get("rootPath") or "")
        local_binary = Path(config.get("binaryPath") or "")
        local_model = Path(config.get("modelPath") or "")
        env = os.environ.copy()
        lib_path = str(local_root / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        session_result = run_local_llm_text_session(
            build_local_deepseek_prompt_v2(report_payload),
            local_root,
            local_binary,
            local_model,
            config,
            env,
        )
        combined = session_result.get("transcript") or ""
        reply_text = session_result.get("reply") or ""
        result_returncode = session_result.get("returncode")
        parsed = extract_json_object_from_text(combined)
        if parsed is None and reply_text:
            parsed = extract_json_object_from_text(reply_text)
        if parsed is None and reply_text.strip():
            parsed = parse_local_deepseek_text_payload(reply_text, report_payload)
        if reply_text.strip():
            result_returncode = 0
        if result_returncode not in (0, None) and parsed is None:
            error_text = f"local llm exited with code {result_returncode}"
            if int(result_returncode) == -9:
                error_text = "local llm exited with code -9 (likely OOM; close chromium or other heavy apps before using local mode)"
            return {
                "ok": False,
                "error": error_text + (f" | server: {server_error}" if server_error else ""),
                "provider": "deepseek-local",
                "transport": "local",
                "providerLabel": "本地 DeepSeek 1.5B",
                "model": local_model.name,
                "checkedRoots": config.get("checkedRoots") or [],
                "rawOutputPreview": compact_report_text(combined, max_chars=260),
            }
        if parsed is None:
            parsed = {
                "summary": compact_report_text(reply_text or combined or "local model returned no parseable JSON", max_chars=120),
                "riskLevel": "medium",
                "actions": [],
                "detailedAnalysis": compact_report_text(reply_text or combined or "local model returned no detailed output", max_chars=260),
                "probabilities": [],
                "componentProfile": {},
            }
        parsed["ok"] = True
        parsed["provider"] = "deepseek-local"
        parsed["transport"] = "local"
        parsed["providerLabel"] = "本地 DeepSeek 1.5B"
        parsed["model"] = local_model.name
        parsed["checkedRoots"] = config.get("checkedRoots") or []
        parsed["rawOutputPreview"] = compact_report_text(reply_text or combined, max_chars=260)
        return parsed
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "provider": "deepseek-local",
            "transport": "local",
            "providerLabel": "本地 DeepSeek 1.5B",
            "model": Path(config.get("modelPath") or LOCAL_LLM_MODEL_NAME).name,
            "checkedRoots": config.get("checkedRoots") or [],
        }


def call_deepseek_summary(report_payload, progress_callback=None, llm_mode="auto"):
    config = load_llm_config()
    local_config = local_llm_config()
    remote_error = None
    if config.get("enabled") and network_endpoint_reachable(config.get("baseUrl", DEEPSEEK_BASE_URL), timeout=2.0):
        if progress_callback:
            progress_callback(progress=42, stage="llm", message="正在调用 DeepSeek API", provider="deepseek-api", providerLabel="DeepSeek API")
        try:
            remote_result = call_remote_deepseek_summary(report_payload, config)
            if remote_result.get("ok"):
                return remote_result
            remote_error = remote_result.get("error")
        except Exception as exc:
            remote_error = str(exc)
    if progress_callback:
        progress_callback(progress=42, stage="llm", message="网络不可用，正在调用本地 DeepSeek 1.5B", provider="deepseek-local", providerLabel="本地 DeepSeek 1.5B")
    local_result = call_local_deepseek_summary(report_payload)
    if local_result.get("ok"):
        return local_result
    if config.get("enabled"):
        if progress_callback:
            progress_callback(progress=52, stage="llm", message="本地模型不可用，回退到 DeepSeek API", provider="deepseek-api", providerLabel="DeepSeek API")
        try:
            fallback_remote = call_remote_deepseek_summary(report_payload, config)
            if fallback_remote.get("ok"):
                return fallback_remote
            fallback_error = fallback_remote.get("error")
        except Exception as exc:
            fallback_error = str(exc)
        return {
            "ok": False,
            "error": f"local failed: {local_result.get('error')} | api failed: {fallback_error or remote_error}",
            "provider": "deepseek-hybrid",
            "transport": "hybrid",
            "providerLabel": "API / 本地切换失败",
        }
    if not local_config.get("enabled"):
        return {
            "ok": False,
            "error": "deepseek api unavailable and local model missing",
            "provider": "deepseek-hybrid",
            "transport": "hybrid",
            "providerLabel": "API / 本地均不可用",
        }
    return local_result


def build_diagnostic_report_payload(analysis, status):
    nearest = deepcopy(analysis.get("nearestComponent")) if analysis else None
    report = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boardId": BOARD_MODEL.get("id"),
        "boardSourceFile": BOARD_MODEL.get("sourceFile"),
        "analysis": deepcopy(analysis),
        "status": deepcopy(status),
        "faultSummary": None,
        "recommendations": [],
    }
    if nearest is None:
        report["faultSummary"] = f"当前未找到与热像准星稳定匹配的器件，热源来源为 {analysis.get('hotspotSource', '--')}。"
        report["recommendations"] = [
            "请重新执行可见光与热成像四孔标定。",
            "请确认热区位于板框内部后再重新生成报告。",
        ]
        return report

    top_probability = analysis.get("probabilities", [{}])[0] if analysis.get("probabilities") else {}
    report["faultSummary"] = (
        f"当前热点主要关联器件为 {nearest.get('ref')}，定位方式为 {nearest.get('matchMode', 'nearest-component')}。"
    )
    report["recommendations"] = [
        f"优先检查 {nearest.get('ref')} 及其周边供电、焊接和相邻热源影响。",
        "对比空载与负载工况热像，确认该热区是否稳定复现。",
        "生成详细诊断 PDF，由大模型结合器件与网表信息进一步分析。",
    ]
    return report


def render_diagnostic_report_pdf(payload):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise RuntimeError(f"reportlab unavailable: {exc}")

    report_dir = reports_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = report_dir / f"diagnostic-report-{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    def pick_chinese_font():
        candidates = [
            ROOT / "vendor" / "NotoSansSC-Regular.ttf",
            ROOT / "vendor" / "NotoSansSC-Bold.ttf",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        ]
        regular_name = "Helvetica"
        bold_name = "Helvetica-Bold"
        regular_path = next((path for path in candidates if path.exists() and "Bold" not in path.name), None)
        bold_path = next((path for path in candidates if path.exists() and "Bold" in path.name), None)
        try:
            if regular_path:
                pdfmetrics.registerFont(TTFont("ReportCnRegular", str(regular_path)))
                regular_name = "ReportCnRegular"
            if bold_path:
                pdfmetrics.registerFont(TTFont("ReportCnBold", str(bold_path)))
                bold_name = "ReportCnBold"
            elif regular_path:
                bold_name = regular_name
        except Exception:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                regular_name = "STSong-Light"
                bold_name = "STSong-Light"
            except Exception:
                regular_name = "Helvetica"
                bold_name = "Helvetica-Bold"
        if regular_name == "Helvetica":
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                regular_name = "STSong-Light"
                bold_name = "STSong-Light"
            except Exception:
                pass
        return regular_name, bold_name

    regular_font, bold_font = pick_chinese_font()
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.fontName = bold_font
    title_style.fontSize = 20
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e2933"),
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13,
        textColor=colors.HexColor("#0b4f6c"),
        spaceAfter=6,
    )

    compact_body = ParagraphStyle("CompactBody", parent=body_style, fontSize=8.8, leading=11)
    compact_section = ParagraphStyle("CompactSection", parent=section_style, fontSize=11, spaceAfter=3)

    analysis = payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    llm_payload = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    profile = llm_payload.get("componentProfile") if isinstance(llm_payload.get("componentProfile"), dict) else {}
    probabilities = llm_payload.get("probabilities") if isinstance(llm_payload.get("probabilities"), list) and llm_payload.get("probabilities") else analysis.get("probabilities", [])
    nets = nearest.get("nets", [])

    story = []
    story.append(Paragraph("PCB 热诊断报告", title_style))
    story.append(Spacer(1, 2 * mm))

    header_rows = [[
        Paragraph(
            f"<b>板卡</b>：{payload.get('boardId', '--')}<br/>"
            f"<b>坐标文件</b>：{payload.get('boardSourceFile', '--')}<br/>"
            f"<b>时间</b>：{payload.get('generatedAt', '--')}",
            compact_body,
        ),
        Paragraph(
            f"<b>热点器件</b>：{nearest.get('ref', '--')}<br/>"
            f"<b>匹配方式</b>：{nearest.get('matchMode', '--')}<br/>"
            f"<b>风险等级</b>：{llm_payload.get('riskLevel', '中')}",
            compact_body,
        ),
    ]]
    header_table = Table(header_rows, colWidths=[96 * mm, 80 * mm])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f9ff")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#bfd2de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 2.5 * mm))

    story.append(Paragraph("诊断摘要", compact_section))
    story.append(Paragraph(str(llm_payload.get("summary") or payload.get("faultSummary") or "--"), compact_body))
    story.append(Spacer(1, 2.5 * mm))

    summary_rows = [
        ["热点图像坐标", str(analysis.get("imagePoint", "--"))],
        ["PCB坐标(mm)", str(analysis.get("boardPointMm", "--"))],
        ["器件封装", str(nearest.get("footprint", "--"))],
        ["器件型号", str(nearest.get("device", "--"))],
        ["器件说明", str(nearest.get("comment", "--"))],
        ["资料提示", str(profile.get("datasheetHints", "--"))],
    ]
    summary_table = Table(summary_rows, colWidths=[34 * mm, 142 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c5d3dc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce6ec")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("FONTNAME", (1, 0), (1, -1), regular_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
            ]
        )
    )
    story.append(Paragraph("定位信息", compact_section))
    story.append(summary_table)
    story.append(Spacer(1, 2.5 * mm))

    fault_rows = [["候选结论", "概率"]]
    for item in probabilities[:4]:
        fault_rows.append([str(item.get("label", "--")), f"{int(item.get('percent', 0))}%"])
    if len(fault_rows) == 1:
        fault_rows.append(["暂无结果", "--"])
    fault_table = Table(fault_rows, colWidths=[136 * mm, 40 * mm])
    fault_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef5ff")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#bfd1e2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe7f2")),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), regular_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )
    story.append(Paragraph("概率判断", compact_section))
    story.append(fault_table)
    story.append(Spacer(1, 2.5 * mm))

    left_column = [
        Paragraph("详细分析", compact_section),
        Paragraph(str(llm_payload.get("detailedAnalysis") or payload.get("faultSummary") or "--"), compact_body),
        Spacer(1, 1.5 * mm),
        Paragraph(f"器件功能推断：{profile.get('probableFunction', '--')}", compact_body),
        Paragraph(f"封装/资料提示：{profile.get('packageNotes', '--')}", compact_body),
    ]
    actions = llm_payload.get("actions") or payload.get("recommendations") or []
    action_text = "<br/>".join([f"{idx + 1}. {item}" for idx, item in enumerate(actions[:3])]) or "1. 暂无建议"
    right_column = [
        Paragraph("建议操作", compact_section),
        Paragraph(action_text, compact_body),
    ]
    if nets:
        net_text = "<br/>".join(
            [f"{item.get('net', '--')} / {item.get('pin', '--')} / {item.get('name', '--')}" for item in nets[:5]]
        )
        right_column.extend([Spacer(1, 1.5 * mm), Paragraph("相关网表", compact_section), Paragraph(net_text, compact_body)])

    detail_table = Table([[left_column, right_column]], colWidths=[104 * mm, 72 * mm])
    detail_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d0dce6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(detail_table)

    doc.build(story)
    return output_path


def call_deepseek_summary(report_payload):
    config = load_llm_config()
    if not config.get("enabled"):
        return {"ok": False, "error": "deepseek api key not configured", "provider": "deepseek"}

    analysis = report_payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    nets = nearest.get("nets") or []
    candidate_summary = nearest.get("candidateSummary") or []
    nets_text = "\n".join(
        f"- 网络: {item.get('net', '--')} / 引脚: {item.get('pin', '--')} / 名称: {item.get('name', '--')}"
        for item in nets[:8]
    ) or "- 暂无网表信息"
    candidate_text = "\n".join(
        f"- {item.get('ref', '--')} / sumHeat={item.get('sumHeat', '--')} / meanHeat={item.get('meanHeat', '--')} / maxHeat={item.get('maxHeat', '--')}"
        for item in candidate_summary[:5]
    ) or "- 暂无候选器件热量排序"
    source_file = BOARD_MODEL.get("sourceFile") or "default-board-model"
    prompt = (
        "你是 PCB 热诊断工程师。请基于以下结构化信息，输出详细、专业、全中文的诊断结果。\n"
        "要求:\n"
        "1. 只返回 JSON，不要输出 markdown。\n"
        "2. JSON 字段必须包含 summary、riskLevel、actions、detailedAnalysis、probabilities、componentProfile。\n"
        "3. summary 控制在 50 字以内，适合首页摘要。\n"
        "4. detailedAnalysis 使用 2 到 4 段中文分析，可以适度详细，但不要空泛重复。\n"
        "5. probabilities 是数组，每项包含 label 和 percent，总和尽量接近 100。\n"
        "6. componentProfile 是对象，包含 probableFunction、packageNotes、datasheetHints 三个字段。\n"
        "7. actions 为中文字符串数组，给 3 条以内建议。\n"
        "8. 允许根据器件型号、封装、常见用途、网表关系和热像位置做合理推断，但不要伪造未经确认的精确结论。\n"
        "9. 如果器件看起来像可检索型号，可以根据常见公开资料推断用途和参数方向，但不要伪造精确参数值。\n\n"
        f"板卡编号: {report_payload.get('boardId', '--')}\n"
        f"坐标来源文件: {source_file}\n"
        f"生成时间: {report_payload.get('generatedAt', '--')}\n"
        f"热源定位来源: {analysis.get('hotspotSource', '--')}\n"
        f"热像准星点: {analysis.get('crosshairPoint', '--')}\n"
        f"热像热点点: {analysis.get('thermalPoint', '--')}\n"
        f"映射后的可见光点: {analysis.get('imagePoint', '--')}\n"
        f"PCB 坐标(mm): {analysis.get('boardPointMm', '--')}\n"
        f"OCR 温度文本: {thermal_ocr.get('text', '--')}\n"
        f"OCR 温度(℃): {thermal_ocr.get('temperatureC', '--')}\n"
        f"最近器件: {nearest.get('ref', '--')}\n"
        f"封装: {nearest.get('footprint', '--')}\n"
        f"型号: {nearest.get('device', '--')}\n"
        f"说明: {nearest.get('comment', '--')}\n"
        f"匹配模式: {nearest.get('matchMode', '--')}\n"
        f"热量分数: {nearest.get('heatScore', '--')}\n"
        f"热点框(热像): {analysis.get('bboxThermal', '--')}\n"
        f"器件图像边界(可见光): {nearest.get('imageBounds', '--')}\n"
        f"候选器件热量排名:\n{candidate_text}\n"
        "网表信息:\n"
        f"{nets_text}\n"
    )
    payload = {
        "model": config.get("model") or DEEPSEEK_DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一名严谨的 PCB 故障诊断助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    url = f"{config.get('baseUrl', DEEPSEEK_BASE_URL)}{DEEPSEEK_CHAT_PATH}"
    response = http_post_json_headers(
        url,
        payload,
        headers={"Authorization": f"Bearer {config['apiKey']}"},
        timeout=35,
    )
    message = (((response.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not message:
        return {"ok": False, "error": "deepseek returned empty content", "provider": "deepseek", "raw": response}
    try:
        parsed = json.loads(message)
    except Exception:
        parsed = {
            "summary": str(message).strip(),
            "riskLevel": "中",
            "actions": [],
            "detailedAnalysis": str(message).strip(),
            "probabilities": [],
            "componentProfile": {},
        }
    parsed["ok"] = True
    parsed["provider"] = "deepseek"
    parsed["model"] = payload["model"]
    return parsed


def call_deepseek_summary(report_payload, progress_callback=None, llm_mode="auto"):
    mode = normalize_llm_mode(llm_mode)
    config = load_llm_config()
    local_config = local_llm_config()
    if mode == "api":
        if not config.get("enabled"):
            return {
                "ok": False,
                "error": "deepseek api key not configured",
                "provider": "deepseek-api",
                "transport": "api",
                "providerLabel": "DeepSeek API",
            }
        if progress_callback:
            progress_callback(progress=42, stage="llm", message="calling DeepSeek API", provider="deepseek-api", providerLabel="DeepSeek API")
        try:
            return call_remote_deepseek_summary(report_payload, config)
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "provider": "deepseek-api",
                "transport": "api",
                "providerLabel": "DeepSeek API",
            }
    if mode == "local":
        if progress_callback:
            progress_callback(progress=42, stage="llm", message="正在调用本地 DeepSeek 1.5B，完整推理可能需要 1 到 3 分钟", provider="deepseek-local", providerLabel="本地 DeepSeek 1.5B")
        local_result = call_local_deepseek_summary(report_payload)
        local_result.setdefault("provider", "deepseek-local")
        local_result.setdefault("transport", "local")
        local_result.setdefault("providerLabel", "本地 DeepSeek 1.5B")
        return local_result
    if config.get("enabled") and network_endpoint_reachable(config.get("baseUrl", DEEPSEEK_BASE_URL), timeout=2.0):
        if progress_callback:
            progress_callback(progress=42, stage="llm", message="正在调用 DeepSeek API", provider="deepseek-api", providerLabel="DeepSeek API")
        remote_result = call_remote_deepseek_summary(report_payload, config)
        if remote_result.get("ok"):
            return remote_result
    if progress_callback:
        progress_callback(progress=42, stage="llm", message="网络不可用，正在调用本地 DeepSeek 1.5B", provider="deepseek-local", providerLabel="本地 DeepSeek 1.5B")
    local_result = call_local_deepseek_summary(report_payload)
    if local_result.get("ok"):
        return local_result
    if config.get("enabled"):
        if progress_callback:
            progress_callback(progress=52, stage="llm", message="本地模型不可用，回退到 DeepSeek API", provider="deepseek-api", providerLabel="DeepSeek API")
        fallback_remote = call_remote_deepseek_summary(report_payload, config)
        if fallback_remote.get("ok"):
            return fallback_remote
        return {
            "ok": False,
            "error": f"local failed: {local_result.get('error')} | api failed: {fallback_remote.get('error')}",
            "provider": "deepseek-hybrid",
            "transport": "hybrid",
            "providerLabel": "API / 本地切换失败",
        }
    if not local_config.get("enabled"):
        return {
            "ok": False,
            "error": "deepseek api unavailable and local model missing",
            "provider": "deepseek-hybrid",
            "transport": "hybrid",
            "providerLabel": "API / 本地均不可用",
        }
    return local_result


def build_diagnostic_report_payload(analysis, status):
    nearest = deepcopy(analysis.get("nearestComponent")) if analysis else None
    thermal_ocr = deepcopy((analysis or {}).get("thermalOcr")) if analysis else None
    component_context = build_component_context(nearest) if nearest else {}
    report = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boardId": BOARD_MODEL.get("id"),
        "boardSourceFile": BOARD_MODEL.get("sourceFile"),
        "analysis": deepcopy(analysis),
        "status": deepcopy(status),
        "componentContext": component_context,
        "faultSummary": None,
        "recommendations": [],
    }
    if nearest is None:
        temp_suffix = ""
        if isinstance(thermal_ocr, dict) and thermal_ocr.get("ok") and thermal_ocr.get("temperatureC") is not None:
            temp_suffix = f" OCR温度约 {thermal_ocr.get('temperatureC')}C。"
        report["faultSummary"] = f"当前未找到与热像准星稳定匹配的器件，热源来源为 {analysis.get('hotspotSource', '--')}。{temp_suffix}"
        report["recommendations"] = [
            "请重新执行可见光与热成像四孔标定。",
            "请确认热区位于板框内部后再重新生成报告。",
        ]
        return report

    temp_suffix = ""
    if isinstance(thermal_ocr, dict) and thermal_ocr.get("ok") and thermal_ocr.get("temperatureC") is not None:
        temp_suffix = f" OCR温度约 {thermal_ocr.get('temperatureC')}C。"
    report["faultSummary"] = (
        f"当前热点主要关联器件为 {nearest.get('ref')}，定位方式为 {nearest.get('matchMode', 'nearest-component')}。{temp_suffix}"
    )
    report["recommendations"] = [
        f"优先检查 {nearest.get('ref')} 及其周边供电、焊接和相邻热源影响。",
        "对比空载与负载工况热像，确认该热区是否稳定复现。",
        "结合坐标文件、器件信息和热像截图生成诊断 PDF 复核结果。",
    ]
    return report


def compact_report_text(text, max_chars=180):
    value = str(text or "--").replace("\r", "\n").strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


TEMPERATURE_WITH_UNIT_RE = re.compile(r"(?P<value>-?\d{2,3}(?:\.\d)?)\s*(?P<unit>°?\s*C|℃|度)", re.IGNORECASE)


def format_temperature_value(value):
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number - round(number)) < 0.05:
        return str(int(round(number)))
    return f"{number:.1f}"


def sanitize_temperature_mentions(text, ocr_temp_c):
    if not text or ocr_temp_c in (None, ""):
        return text
    try:
        target = float(ocr_temp_c)
    except Exception:
        return text

    def replace_match(match):
        try:
            detected = float(match.group("value"))
        except Exception:
            return match.group(0)
        if abs(detected - target) <= 5.0:
            return match.group(0)
        unit = match.group("unit")
        normalized_unit = "℃" if "℃" in unit else ("度" if "度" in unit else "C")
        return f"{format_temperature_value(target)}{normalized_unit}"

    return TEMPERATURE_WITH_UNIT_RE.sub(replace_match, str(text))


def sanitize_llm_temperature_fields(llm_payload, report_payload):
    if not isinstance(llm_payload, dict):
        return llm_payload
    analysis = (report_payload or {}).get("analysis") or {}
    thermal_ocr = analysis.get("thermalOcr") or {}
    ocr_temp = thermal_ocr.get("temperatureC")
    if ocr_temp in (None, ""):
        return llm_payload

    cleaned = deepcopy(llm_payload)
    for key in ("summary", "detailedAnalysis"):
        if key in cleaned:
            cleaned[key] = sanitize_temperature_mentions(cleaned.get(key), ocr_temp)
    if isinstance(cleaned.get("actions"), list):
        cleaned["actions"] = [sanitize_temperature_mentions(item, ocr_temp) for item in cleaned.get("actions", [])]
    return cleaned


def render_diagnostic_report_pdf(payload):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise RuntimeError(f"reportlab unavailable: {exc}")

    report_dir = reports_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = report_dir / f"diagnostic-report-{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=11 * mm,
        bottomMargin=11 * mm,
    )

    def pick_chinese_font():
        candidates = [
            ROOT / "vendor" / "msyh.ttc",
            ROOT / "vendor" / "msyhbd.ttc",
            ROOT / "vendor" / "NotoSansSC-Regular.ttf",
            ROOT / "vendor" / "NotoSansSC-Bold.ttf",
            Path("/usr/share/fonts/truetype/msttcorefonts/microsoftyahei.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/msyh.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        ]
        regular_name = "Helvetica"
        bold_name = "Helvetica-Bold"
        regular_path = next((path for path in candidates if path.exists() and path.name.lower() not in ("msyhbd.ttc",) and "Bold" not in path.name), None)
        bold_path = next((path for path in candidates if path.exists() and (path.name.lower() in ("msyhbd.ttc",) or "Bold" in path.name)), None)
        try:
            if regular_path:
                pdfmetrics.registerFont(TTFont("ReportCnRegularCompact", str(regular_path)))
                regular_name = "ReportCnRegularCompact"
            if bold_path:
                pdfmetrics.registerFont(TTFont("ReportCnBoldCompact", str(bold_path)))
                bold_name = "ReportCnBoldCompact"
            elif regular_path:
                bold_name = regular_name
        except Exception:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                regular_name = "STSong-Light"
                bold_name = "STSong-Light"
            except Exception:
                regular_name = "Helvetica"
                bold_name = "Helvetica-Bold"
        if regular_name == "Helvetica":
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                regular_name = "STSong-Light"
                bold_name = "STSong-Light"
            except Exception:
                pass
        return regular_name, bold_name

    regular_font, bold_font = pick_chinese_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CompactTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=17,
        leading=20,
        textColor=colors.HexColor("#0f3d91"),
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "CompactBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=8.4,
        leading=10.5,
        textColor=colors.HexColor("#16324f"),
    )
    small_style = ParagraphStyle(
        "CompactSmall",
        parent=body_style,
        fontSize=7.6,
        leading=9.2,
        textColor=colors.HexColor("#4a6380"),
    )
    section_style = ParagraphStyle(
        "CompactSection",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=10.6,
        leading=12.4,
        textColor=colors.HexColor("#1457b5"),
        spaceAfter=3,
        spaceBefore=1,
    )

    analysis = payload.get("analysis") or {}
    nearest = analysis.get("nearestComponent") or {}
    llm_payload = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    profile = llm_payload.get("componentProfile") if isinstance(llm_payload.get("componentProfile"), dict) else {}
    probabilities = llm_payload.get("probabilities") if isinstance(llm_payload.get("probabilities"), list) and llm_payload.get("probabilities") else analysis.get("probabilities", [])
    nets = nearest.get("nets", [])
    thermal_ocr = analysis.get("thermalOcr") or {}
    report_images = payload.get("reportImages") if isinstance(payload.get("reportImages"), dict) else {}

    ocr_temp_text = "--"
    if thermal_ocr.get("ok") and thermal_ocr.get("temperatureC") is not None:
        ocr_temp_text = f"{thermal_ocr.get('temperatureC')} C"

    story = [
        Paragraph("PCB 热诊断报告", title_style),
        Spacer(1, 1.2 * mm),
    ]

    header_rows = [[
        Paragraph(
            f"<b>板卡</b>: {payload.get('boardId', '--')}<br/>"
            f"<b>坐标文件</b>: {payload.get('boardSourceFile', '--')}<br/>"
            f"<b>时间</b>: {payload.get('generatedAt', '--')}",
            body_style,
        ),
        Paragraph(
            f"<b>热点器件</b>: {nearest.get('ref', '--')}<br/>"
            f"<b>风险等级</b>: {llm_payload.get('riskLevel', '中')}<br/>"
            f"<b>OCR温度</b>: {ocr_temp_text}",
            body_style,
        ),
    ]]
    header_table = Table(header_rows, colWidths=[102 * mm, 72 * mm])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f9ff")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#8fb6f2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 1.6 * mm))

    story.append(Paragraph("诊断摘要", section_style))
    story.append(Paragraph(compact_report_text(llm_payload.get("summary") or payload.get("faultSummary") or "--", 78), body_style))
    story.append(Spacer(1, 1.6 * mm))

    image_cells = []
    image_captions = []
    for key in ("fusionFocus", "thermalFocus"):
        item = report_images.get(key) if isinstance(report_images, dict) else None
        image_path = Path(str(item.get("path"))) if isinstance(item, dict) and item.get("path") else None
        if image_path is None or not image_path.exists():
            continue
        report_image = Image(str(image_path), width=84 * mm, height=48 * mm)
        image_cells.append(report_image)
        image_captions.append(Paragraph(str(item.get("caption") or "--"), small_style))
    if image_cells:
        while len(image_cells) < 2:
            image_cells.append(Paragraph("", small_style))
            image_captions.append(Paragraph("", small_style))
        image_table = Table([image_cells, image_captions], colWidths=[86 * mm, 86 * mm])
        image_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                    ("TOPPADDING", (0, 1), (-1, 1), 0),
                ]
            )
        )
        story.append(Paragraph("关键截图", section_style))
        story.append(image_table)
        story.append(Spacer(1, 1.8 * mm))

    summary_rows = [
        ["热像热点坐标", str(analysis.get("thermalPoint", "--"))],
        ["可见光坐标", str(analysis.get("imagePoint", "--"))],
        ["PCB坐标(mm)", str(analysis.get("boardPointMm", "--"))],
        ["OCR温度文本", str(thermal_ocr.get("text", "--"))],
        ["器件封装 / 型号", f"{nearest.get('footprint', '--')} / {nearest.get('device', '--')}"],
        ["器件说明", str(nearest.get("comment", "--"))],
    ]
    summary_table = Table(summary_rows, colWidths=[35 * mm, 139 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#c5d3dc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e5fb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("FONTNAME", (1, 0), (1, -1), regular_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1457b5")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f9ff")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Paragraph("定位信息", section_style))
    story.append(summary_table)
    story.append(Spacer(1, 1.8 * mm))

    fault_rows = [["候选结论", "概率"]]
    for item in probabilities[:3]:
        fault_rows.append([str(item.get("label", "--")), f"{int(item.get('percent', 0))}%"])
    if len(fault_rows) == 1:
        fault_rows.append(["暂无结果", "--"])
    fault_table = Table(fault_rows, colWidths=[140 * mm, 34 * mm])
    fault_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e9ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d91")),
                ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#8fb6f2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e5fb")),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), regular_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Paragraph("AI 概率判断", section_style))
    story.append(fault_table)
    story.append(Spacer(1, 1.8 * mm))

    analysis_text = compact_report_text(llm_payload.get("detailedAnalysis") or payload.get("faultSummary") or "--", 210).replace("\n", "<br/>")
    action_items = llm_payload.get("actions") or payload.get("recommendations") or []
    action_text = "<br/>".join([f"{idx + 1}. {compact_report_text(item, 42)}" for idx, item in enumerate(action_items[:3])]) or "1. 暂无建议"
    net_text = "<br/>".join(
        [f"{item.get('net', '--')} / {item.get('pin', '--')} / {item.get('name', '--')}" for item in nets[:4]]
    ) or "--"
    profile_text = (
        f"功能推断: {compact_report_text(profile.get('probableFunction', '--'), 46)}<br/>"
        f"封装提示: {compact_report_text(profile.get('packageNotes', '--'), 46)}<br/>"
        f"资料提示: {compact_report_text(profile.get('datasheetHints', '--'), 46)}"
    )

    left_column = [
        Paragraph("AI 分析", section_style),
        Paragraph(analysis_text, body_style),
        Spacer(1, 1.0 * mm),
        Paragraph(profile_text, small_style),
    ]
    right_column = [
        Paragraph("建议与网表", section_style),
        Paragraph(action_text, body_style),
        Spacer(1, 1.0 * mm),
        Paragraph(f"相关网表:<br/>{net_text}", small_style),
    ]

    detail_table = Table([[left_column, right_column]], colWidths=[108 * mm, 66 * mm])
    detail_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ebcf0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(detail_table)

    doc.build(story)
    return output_path


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def rotate_variant(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def most_common_rotation(rotations):
    counts = {}
    for rotation in rotations:
        counts[rotation] = counts.get(rotation, 0) + 1
    return max(counts, key=counts.get)


def quad_aspect(quad):
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    return width / max(height, 1e-6)


def quad_center(quad):
    pts = np.asarray(quad, dtype=np.float32)
    return pts.mean(axis=0)


def expanded_quad_bounds(quad, frame_shape, margin_ratio=0.12):
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2):
        return (0, 0, frame_shape[1], frame_shape[0])
    min_x = float(np.min(quad[:, 0]))
    min_y = float(np.min(quad[:, 1]))
    max_x = float(np.max(quad[:, 0]))
    max_y = float(np.max(quad[:, 1]))
    pad_x = max(12.0, (max_x - min_x) * float(margin_ratio))
    pad_y = max(12.0, (max_y - min_y) * float(margin_ratio))
    x1 = max(0, int(np.floor(min_x - pad_x)))
    y1 = max(0, int(np.floor(min_y - pad_y)))
    x2 = min(frame_shape[1], int(np.ceil(max_x + pad_x)))
    y2 = min(frame_shape[0], int(np.ceil(max_y + pad_y)))
    return x1, y1, x2, y2


def select_outline_component(mask, coarse_quad):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    coarse_center = quad_center(coarse_quad)
    coarse_area = max(abs(cv2.contourArea(np.asarray(coarse_quad, dtype=np.float32))), 1.0)
    chosen = None
    best_score = -1e9
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 8.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
        distance = float(np.linalg.norm(center - coarse_center))
        score = float(area / coarse_area) * 8.0 - distance / max(np.sqrt(coarse_area), 1.0)
        if score > best_score:
            best_score = score
            chosen = contour
    if chosen is None:
        return None
    output = np.zeros_like(mask)
    cv2.drawContours(output, [chosen], -1, 255, -1)
    return output


def build_visible_outline_mask(frame, coarse_quad):
    x1, y1, x2, y2 = expanded_quad_bounds(coarse_quad, frame.shape[:2], margin_ratio=0.14)
    if x2 <= x1 + 24 or y2 <= y1 + 24:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 22) & (hsv[:, :, 2] >= 20))
        | ((hsv[:, :, 1] >= 28) & (hsv[:, :, 2] <= 165))
    ).astype(np.uint8) * 255
    red = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 80))
    mask[red] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)), iterations=2)
    selected = select_outline_component(mask, np.asarray(coarse_quad, dtype=np.float32) - np.array([x1, y1], dtype=np.float32))
    if selected is None:
        return None
    output = np.zeros(frame.shape[:2], dtype=np.uint8)
    output[y1:y2, x1:x2] = selected
    return output


def build_thermal_outline_mask(frame, coarse_quad):
    x1, y1, x2, y2 = expanded_quad_bounds(coarse_quad, frame.shape[:2], margin_ratio=0.18)
    if x2 <= x1 + 20 or y2 <= y1 + 20:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 45, 35], dtype=np.uint8),
        np.array([75, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    selected = select_outline_component(mask, np.asarray(coarse_quad, dtype=np.float32) - np.array([x1, y1], dtype=np.float32))
    if selected is None:
        return None
    output = np.zeros(frame.shape[:2], dtype=np.uint8)
    output[y1:y2, x1:x2] = selected
    return output


def binary_mask_overlap_score(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    union = float(np.count_nonzero(a | b))
    if union <= 1.0:
        return 0.0
    inter = float(np.count_nonzero(a & b))
    return inter / union


def refine_homography_by_board_outline(visible, thermal, initial_matrix, visible_quad, thermal_quad):
    if initial_matrix is None or visible_quad is None or thermal_quad is None:
        return initial_matrix
    matrix = np.asarray(initial_matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        return initial_matrix
    visible_mask = build_visible_outline_mask(visible, visible_quad)
    thermal_mask = build_thermal_outline_mask(thermal, thermal_quad)
    if visible_mask is None or thermal_mask is None:
        return matrix

    x1, y1, x2, y2 = expanded_quad_bounds(visible_quad, visible.shape[:2], margin_ratio=0.08)
    if x2 <= x1 + 40 or y2 <= y1 + 40:
        return matrix

    warped_before = cv2.warpPerspective(thermal_mask, matrix, (visible.shape[1], visible.shape[0]))
    visible_crop = visible_mask[y1:y2, x1:x2]
    thermal_crop = warped_before[y1:y2, x1:x2]
    if np.count_nonzero(visible_crop) < 300 or np.count_nonzero(thermal_crop) < 300:
        return matrix

    template = cv2.GaussianBlur(visible_crop.astype(np.float32) / 255.0, (5, 5), 0)
    moving = cv2.GaussianBlur(thermal_crop.astype(np.float32) / 255.0, (5, 5), 0)
    warp_affine = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        _, warp_affine = cv2.findTransformECC(
            template,
            moving,
            warp_affine,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            1,
        )
    except cv2.error:
        return matrix

    linear = warp_affine[:, :2]
    det = float(np.linalg.det(linear))
    shift = float(np.linalg.norm(warp_affine[:, 2]))
    off_diag = float(max(abs(float(linear[0, 1])), abs(float(linear[1, 0]))))
    if det < 0.85 or det > 1.18 or shift > max(x2 - x1, y2 - y1) * 0.10 or off_diag > 0.18:
        return matrix

    delta = np.eye(3, dtype=np.float32)
    delta[:2, :] = warp_affine
    translate = np.array([[1.0, 0.0, float(x1)], [0.0, 1.0, float(y1)], [0.0, 0.0, 1.0]], dtype=np.float32)
    translate_inv = np.array([[1.0, 0.0, -float(x1)], [0.0, 1.0, -float(y1)], [0.0, 0.0, 1.0]], dtype=np.float32)
    refined = translate @ delta @ translate_inv @ matrix

    warped_after = cv2.warpPerspective(thermal_mask, refined, (visible.shape[1], visible.shape[0]))
    score_before = binary_mask_overlap_score(visible_crop, thermal_crop)
    score_after = binary_mask_overlap_score(visible_mask[y1:y2, x1:x2], warped_after[y1:y2, x1:x2])
    if score_after < score_before + 0.006:
        return matrix
    return refined.astype(np.float32)


def quad_score_against_model(quad, frame_shape):
    ordered = order_quad(quad)
    area = abs(cv2.contourArea(ordered))
    frame_area = frame_shape[0] * frame_shape[1]
    area_ratio = area / max(frame_area, 1.0)
    aspect = quad_aspect(ordered)
    aspect_penalty = abs(np.log(max(aspect, 1e-6) / BOARD_MODEL["aspect"]))
    center = quad_center(ordered)
    cx = center[0] / max(frame_shape[1], 1.0)
    cy = center[1] / max(frame_shape[0], 1.0)
    center_penalty = abs(cx - 0.61) * 0.8 + abs(cy - 0.60) * 1.1
    return area_ratio * 10.0 - aspect_penalty * 4.0 - center_penalty


def refine_visible_quad_by_pcb_edges(frame, coarse_quad):
    coarse_quad = np.asarray(coarse_quad, dtype=np.float32)
    height, width = frame.shape[:2]
    x1 = max(0, int(np.min(coarse_quad[:, 0]) - 80))
    y1 = max(0, int(np.min(coarse_quad[:, 1]) - 80))
    x2 = min(width, int(np.max(coarse_quad[:, 0]) + 80))
    y2 = min(height, int(np.max(coarse_quad[:, 1]) + 80))
    if x2 <= x1 + 40 or y2 <= y1 + 40:
        return coarse_quad

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    board_mask = (
        ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25))
        | ((hsv[:, :, 1] >= 35) & (hsv[:, :, 2] <= 155))
    ).astype(np.uint8) * 255

    red = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 80))
    board_mask[red] = 0
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)), iterations=2)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return coarse_quad

    candidates = []
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.006 or area > frame_area * 0.08:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] += x1
        box[:, 1] += y1
        ordered = order_quad(box)
        aspect = quad_aspect(ordered)
        if aspect < 1.25 or aspect > 2.25:
            continue
        candidates.append((quad_score_against_model(ordered, frame.shape[:2]) + area / max(frame_area, 1.0) * 15.0, ordered))

    if not candidates:
        return coarse_quad
    refined = max(candidates, key=lambda item: item[0])[1]
    center_shift = np.linalg.norm(quad_center(refined) - quad_center(coarse_quad))
    coarse_width = max(np.linalg.norm(coarse_quad[1] - coarse_quad[0]), np.linalg.norm(coarse_quad[2] - coarse_quad[3]))
    if center_shift > coarse_width * 0.22:
        return coarse_quad
    return refined.astype(np.float32)


def refine_thermal_quad_by_pcb_edges(frame, coarse_quad):
    coarse_quad = np.asarray(coarse_quad, dtype=np.float32)
    refined = find_thermal_pcb_quad(frame)
    if refined is None:
        return coarse_quad
    refined = order_quad(np.asarray(refined, dtype=np.float32))
    coarse_center = quad_center(coarse_quad)
    refined_center = quad_center(refined)
    center_shift = np.linalg.norm(refined_center - coarse_center)
    coarse_width = max(np.linalg.norm(coarse_quad[1] - coarse_quad[0]), np.linalg.norm(coarse_quad[2] - coarse_quad[3]))
    coarse_area = abs(cv2.contourArea(coarse_quad.astype(np.float32)))
    refined_area = abs(cv2.contourArea(refined.astype(np.float32)))
    if coarse_width <= 1e-6 or coarse_area <= 1e-6:
        return coarse_quad
    area_ratio = refined_area / coarse_area
    if center_shift > coarse_width * 0.24:
        return coarse_quad
    if area_ratio < 0.72 or area_ratio > 1.32:
        return coarse_quad
    return refined.astype(np.float32)


def select_best_thermal_quad(frame, visible_quad):
    target_aspect = quad_aspect(visible_quad)
    best_rotation = "none"
    best_quad = None
    best_score = -1.0

    for rotation in ("none", "cw", "ccw", "180"):
        rotated = rotate_variant(frame, rotation)
        quad = find_board_quad(rotated, prefer_color=False)
        if quad is None:
            continue
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        aspect = quad_aspect(quad)
        penalty = abs(np.log(max(aspect, 1e-6) / max(target_aspect, 1e-6)))
        score = area / (1.0 + penalty * 4.0)
        if score > best_score:
            best_score = score
            best_rotation = rotation
            best_quad = quad

    return best_rotation, best_quad


def find_board_quad(frame, prefer_color=True):
    if prefer_color:
        pcb_quad = find_visible_pcb_quad(frame)
        return pcb_quad

    thermal_quad = find_thermal_pcb_quad(frame)
    if thermal_quad is not None:
        return thermal_quad

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    # Fallback for thermal pseudo-color frames where the board may be a bright blob.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    frame_area = frame.shape[0] * frame.shape[1]
    best_quad = None
    best_score = 0

    for contour in candidates:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.05 or area > frame_area * 0.96:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)

        ordered = order_quad(quad)
        width_a = np.linalg.norm(ordered[2] - ordered[3])
        width_b = np.linalg.norm(ordered[1] - ordered[0])
        height_a = np.linalg.norm(ordered[1] - ordered[2])
        height_b = np.linalg.norm(ordered[0] - ordered[3])
        width = max(width_a, width_b)
        height = max(height_a, height_b)
        if width < 20 or height < 20:
            continue
        aspect = width / max(height, 1)
        if aspect < 0.35 or aspect > 3.5:
            continue
        score = area
        if score > best_score:
            best_score = score
            best_quad = ordered

    return best_quad


def find_visible_pcb_quad(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]

    roi = np.zeros((height, width), dtype=np.uint8)
    roi[int(height * 0.30) : int(height * 0.84), int(width * 0.38) : int(width * 0.92)] = 255

    board_like = cv2.inRange(
        hsv,
        np.array([0, 40, 0], dtype=np.uint8),
        np.array([179, 255, 220], dtype=np.uint8),
    )
    board_like = cv2.bitwise_and(board_like, roi)
    red_marks = cv2.inRange(
        hsv,
        np.array([0, 70, 80], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8),
    )
    red_marks = cv2.bitwise_or(
        red_marks,
        cv2.inRange(
            hsv,
            np.array([165, 70, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        ),
    )
    board_like[red_marks > 0] = 0
    board_like = cv2.morphologyEx(board_like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    board_like = cv2.morphologyEx(
        board_like,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 13)),
        iterations=1,
    )

    contours, _ = cv2.findContours(board_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_candidates = []
    frame_area = height * width
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.025 or area > frame_area * 0.14:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > width * 0.34 or bh > height * 0.32:
            continue
        if bw < 260 or bh < 170:
            continue
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        if not (width * 0.45 <= center_x <= width * 0.82 and height * 0.40 <= center_y <= height * 0.78):
            continue
        board_h = bw / BOARD_MODEL["aspect"]
        y1 = max(0, y - board_h * 0.04)
        x1 = max(0, x - bw * 0.025)
        x2 = min(width - 1, x + bw * 1.025)
        y2 = min(height - 1, y1 + board_h * 1.08)
        quad = rect_to_quad(x1, y1, x2, y2)
        body_candidates.append((quad_score_against_model(quad, frame.shape[:2]) + area / frame_area * 20.0, quad))

    if body_candidates:
        body_candidates.sort(key=lambda item: item[0], reverse=True)
        return body_candidates[0][1]

    green = cv2.inRange(
        hsv,
        np.array([35, 35, 25], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    green = cv2.bitwise_and(green, roi)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 19)),
        iterations=2,
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.bitwise_and(edges, roi)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    candidates = []
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.006 or area > frame_area * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 240 or bh < 140 or bw > width * 0.32 or bh > height * 0.30:
            continue
        body_w = max(bw * 1.72, bh * BOARD_MODEL["aspect"] * 1.18)
        body_h = body_w / BOARD_MODEL["aspect"]
        x1 = max(0, x - body_w * 0.08)
        y1 = max(0, y - body_h * 0.08)
        x2 = min(width - 1, x1 + body_w)
        y2 = min(height - 1, y1 + body_h)
        if x2 - x1 > width * 0.36 or y2 - y1 > height * 0.34:
            continue
        quad = rect_to_quad(x1, y1, x2, y2)
        candidates.append((quad_score_against_model(quad, frame.shape[:2]) + area / frame_area * 25.0, quad))

    for mask, expand_x, expand_y in ((green, 0.08, 0.10), (edges, 0.02, 0.03)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < height * width * 0.008 or area > height * width * 0.20:
                continue
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)
            ordered = order_quad(quad)
            aspect = quad_aspect(ordered)
            if aspect < 1.25 or aspect > 2.4:
                continue
            x, y, bw, bh = cv2.boundingRect(ordered.astype(np.int32))
            if bw > width * 0.36 or bh > height * 0.34:
                continue
            x1 = max(0, x - bw * expand_x)
            y1 = max(0, y - bh * expand_y)
            x2 = min(width - 1, x + bw * (1.0 + expand_x))
            y2 = min(height - 1, y + bh * (1.0 + expand_y))
            expanded = rect_to_quad(x1, y1, x2, y2)
            candidates.append((quad_score_against_model(expanded, frame.shape[:2]), expanded))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_thermal_pcb_quad(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(
        hsv,
        np.array([0, 60, 60], dtype=np.uint8),
        np.array([50, 255, 255], dtype=np.uint8),
    )
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(warm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = frame.shape[:2]
    frame_area = height * width
    best = None
    best_score = -1e9
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.05 or area > frame_area * 0.75:
            continue
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype(np.float32)
        ordered = order_quad(quad)
        aspect = quad_aspect(ordered)
        if aspect < 1.2 or aspect > 2.2:
            continue
        x, y, bw, bh = cv2.boundingRect(ordered.astype(np.int32))
        pad_x = bw * 0.10
        pad_y = bh * 0.08
        expanded = rect_to_quad(
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(width - 1, x + bw + pad_x),
            min(height - 1, y + bh + pad_y),
        )
        score = quad_score_against_model(expanded, frame.shape[:2])
        score += area / max(frame_area, 1.0) * 3.0
        if score > best_score:
            best = expanded
            best_score = score

    return best


def rect_to_quad(x1, y1, x2, y2):
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def expand_pcb_rect_size(width, height):
    # The blue solder-mask region excludes connectors and dark component edges.
    # Expand mostly along the board's long axis so the locked quad tracks the
    # physical PCB rather than only the visible blue color blob.
    if width >= height:
        return (width * 1.36, height * 1.08)
    return (width * 1.08, height * 1.36)


def find_board_quad_by_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # PCB solder mask is usually green/cyan and sufficiently saturated compared
    # with the table/background. This is intentionally broad for live lighting.
    lower = np.array([35, 35, 25], dtype=np.uint8)
    upper = np.array([105, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.05 or area > frame_area * 0.96:
            continue
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype(np.float32)
        ordered = order_quad(quad)
        width = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
        height = max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))
        aspect = width / max(height, 1)
        if aspect < 0.45 or aspect > 3.0:
            continue
        if area > best_area:
            best_area = area
            best = ordered

    return best


fusion_worker = FusionWorker(visible_worker, thermal_worker)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._handle_status()
            return
        if path == "/api/report-progress":
            self._handle_report_progress()
            return
        if path == "/api/settings":
            self._handle_settings(parsed.query)
            return
        if path == "/api/snapshot.jpg":
            self._handle_snapshot()
            return
        if path == "/api/debug-holes.jpg":
            self._handle_debug_holes()
            return
        if path == "/api/visible.mjpg":
            self._handle_stream(visible_worker)
            return
        if path == "/api/thermal.mjpg":
            self._handle_stream(thermal_worker)
            return
        if path == "/api/fusion.mjpg":
            self._handle_stream(fusion_worker)
            return
        if path == "/api/recalibrate":
            self._handle_recalibrate()
            return
        if path == "/api/run-yolo-obb":
            self._handle_run_yolo_obb()
            return
        if path == "/api/generate-report":
            self._handle_generate_report(parsed.query)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/manual-board-pose":
            self._handle_manual_board_pose()
            return
        if parsed.path == "/api/board-model":
            self._handle_board_model_upload()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _handle_status(self):
        payload = {
            "visible": visible_worker.get_status(),
            "thermal": thermal_worker.get_status(),
            "fusion": fusion_worker.get_status(),
            "reportTask": fusion_worker.get_report_state(),
            "serverTime": time.time(),
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_settings(self, query):
        params = parse_qs(query)
        alpha = None
        manual_adjust = {}
        reset_manual = False
        save_manual = False
        if "alpha" in params and params["alpha"]:
            try:
                alpha = float(params["alpha"][0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid alpha")
                return
        for key in ("dx", "dy", "scaleX", "scaleY", "sourceLeft", "sourceRight", "sourceTop", "sourceBottom"):
            if key in params and params[key]:
                try:
                    manual_adjust[key] = float(params[key][0])
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, f"invalid {key}")
                    return
        if "resetManual" in params and params["resetManual"]:
            reset_manual = params["resetManual"][0] in ("1", "true", "yes")
        if "saveManual" in params and params["saveManual"]:
            save_manual = params["saveManual"][0] in ("1", "true", "yes")
        if "clearBoardPose" in params and params["clearBoardPose"]:
            self._write_json(fusion_worker.clear_manual_board_pose())
            return
        if "clearThermalBoardPose" in params and params["clearThermalBoardPose"]:
            self._write_json(fusion_worker.clear_manual_thermal_pose())
            return
        if "clearRuntimeBoardModel" in params and params["clearRuntimeBoardModel"]:
            self._write_json(fusion_worker.clear_runtime_board_model())
            return
        payload = fusion_worker.update_settings(
            alpha=alpha,
            manual_adjust=manual_adjust if manual_adjust else None,
            reset_manual=reset_manual,
            save_manual=save_manual,
        )
        self._write_json(payload)

    def _handle_recalibrate(self):
        payload = fusion_worker.request_recalibrate()
        self._write_json(payload)

    def _handle_run_yolo_obb(self):
        payload = fusion_worker.run_yolo_obb_once()
        self._write_json(payload)

    def _handle_generate_report(self, query):
        params = parse_qs(query)
        llm_api_url = params.get("llmApiUrl", [None])[0]
        llm_mode = normalize_llm_mode(params.get("llmMode", ["auto"])[0])
        sync = params.get("sync", ["0"])[0] in ("1", "true", "yes")
        if sync:
            payload = fusion_worker.generate_diagnostic_report(llm_api_url=llm_api_url, llm_mode=llm_mode)
        else:
            payload = fusion_worker.start_diagnostic_report(llm_api_url=llm_api_url, llm_mode=llm_mode)
        self._write_json(payload)

    def _handle_report_progress(self):
        self._write_json({"ok": True, "task": fusion_worker.get_report_state()})

    def _handle_manual_board_pose(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            points = payload.get("points", [])
            target = str(payload.get("target", "visible")).strip().lower()
            if len(points) != 4:
                self.send_error(HTTPStatus.BAD_REQUEST, "expected four points")
                return
            frame = thermal_worker.get_frame() if target == "thermal" else visible_worker.get_frame()
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, f"{target} frame unavailable")
                return
            height, width = frame.shape[:2]
            image_points = []
            for point in points:
                if "xNorm" in point and "yNorm" in point:
                    image_points.append([float(point["xNorm"]) * width, float(point["yNorm"]) * height])
                else:
                    image_points.append([float(point["x"]), float(point["y"])])
            if target == "thermal":
                result = fusion_worker.set_manual_thermal_pose(image_points)
            else:
                result = fusion_worker.set_manual_board_pose(image_points)
            self._write_json(result)
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_board_model_upload(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model = build_runtime_board_model(payload)
            result = fusion_worker.apply_runtime_board_model(model)
            self._write_json(result)
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _write_json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_snapshot(self):
        frame = fusion_worker.get_jpeg()
        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "fusion frame unavailable")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f'attachment; filename="fusion-{int(time.time())}.jpg"')
        self.end_headers()
        self.wfile.write(frame)

    def _handle_debug_holes(self):
        frame = fusion_worker.get_debug_jpeg()
        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "debug frame unavailable")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(frame)

    def _handle_stream(self, worker):
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = worker.get_jpeg()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(getattr(worker, "stream_interval", 0.08))
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


def main():
    visible_worker.start()
    thermal_worker.start()
    fusion_worker.start()

    server = ReusableThreadingHTTPServer((HOST, PORT), DashboardHandler)

    def shutdown(*_args):
        visible_worker.stop()
        thermal_worker.stop()
        fusion_worker.stop()
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"dashboard server running at http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
