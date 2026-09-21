import os
import re
import io
import base64
import threading
from pathlib import Path
import requests

import numpy as np
import cv2
import rasterio
from flask import Blueprint, request, jsonify
from PIL import Image, ImageDraw

# Create the Blueprint
vlm_bp = Blueprint('vlm_bp', __name__, url_prefix='/api/vlm')

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_MODEL = os.getenv(
    "SATQUERY_BASE_MODEL",
    "google/paligemma2-3b-pt-224",
)

ADAPTER_PATH = os.getenv(
    "SATQUERY_ADAPTER",
    r"F:\single-data\satquery_paligemma_lora_4gpu_1k\checkpoint-2400",
)

# Remote Colab-hosted VLM server (used when the local machine has no GPU).
# If this env var is set, /predict forwards requests there instead of
# trying to load the 3B model locally.
REMOTE_VLM_URL = os.getenv("SATQUERY_VLM_REMOTE_URL", "").rstrip("/")

DEVICE = "cuda:0"
MAX_NEW_TOKENS = int(os.getenv("SATQUERY_MAX_NEW_TOKENS", "128"))
MAX_IMAGE_SIZE = 4096

_model = None
_processor = None
_model_lock = threading.Lock()


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
def load_model():
    global _model, _processor

    if _model is not None:
        return _model, _processor

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        PaliGemmaForConditionalGeneration,
    )
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Start this app on a machine with an NVIDIA GPU.")

    adapter = Path(ADAPTER_PATH)
    if not adapter.exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found:\n{adapter}\n\nSet SATQUERY_ADAPTER to your checkpoint directory."
        )

    print("=" * 80)
    print("🚀 Initializing SatQuery PaliGemma 2 VLM Blueprint...")
    print(f"Base model : {BASE_MODEL}")
    print(f"Adapter    : {adapter}")
    print("=" * 80)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    _processor = AutoProcessor.from_pretrained(BASE_MODEL)

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": DEVICE},
    )

    _model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
    )

    _model.eval()
    _model.config.use_cache = True

    print("✅ VLM Model + LoRA adapter loaded successfully.")
    return _model, _processor


# ---------------------------------------------------------------------
# INFERENCE LOGIC
# ---------------------------------------------------------------------
def generate_answer(image: Image.Image, prompt: str):
    import torch
    model, processor = load_model()

    image = image.convert("RGB")

    if max(image.size) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(DEVICE)

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    new_tokens = generated_ids[0, input_length:]
    raw_text = processor.decode(new_tokens, skip_special_tokens=False).strip()
    clean_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return clean_text, raw_text


# ---------------------------------------------------------------------
# GROUNDING / LOC TOKEN PARSER
# ---------------------------------------------------------------------
LOC_RE = re.compile(r"<loc(\d{1,4})>")

def parse_loc_boxes(text, width, height):
    """
    PaliGemma native box order: <locYmin><locXmin><locYmax><locXmax>
    Coordinates are normalized to 0..1023.
    """
    values = [int(x) for x in LOC_RE.findall(text)]
    boxes = []

    for i in range(0, len(values) - 3, 4):
        ymin, xmin, ymax, xmax = values[i:i + 4]

        xmin_px = int(round(xmin / 1023.0 * width))
        ymin_px = int(round(ymin / 1023.0 * height))
        xmax_px = int(round(xmax / 1023.0 * width))
        ymax_px = int(round(ymax / 1023.0 * height))

        xmin_px = max(0, min(width - 1, xmin_px))
        ymin_px = max(0, min(height - 1, ymin_px))
        xmax_px = max(0, min(width - 1, xmax_px))
        ymax_px = max(0, min(height - 1, ymax_px))

        if xmax_px > xmin_px and ymax_px > ymin_px:
            boxes.append({"xmin": xmin_px, "ymin": ymin_px, "xmax": xmax_px, "ymax": ymax_px})

    return boxes


def draw_boxes(image, boxes):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    line_width = max(3, min(img.size) // 150)

    for idx, box in enumerate(boxes, start=1):
        xy = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        draw.rectangle(xy, outline="red", width=line_width)
        text_y = max(0, box["ymin"] - 22)
        draw.text((box["xmin"], text_y), f"object {idx}", fill="red")

    return img


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


# ---------------------------------------------------------------------
# GEOTIFF / STANDARD IMAGE BYTE STREAM LOADER
# ---------------------------------------------------------------------
def load_uploaded_image(file_storage) -> Image.Image:
    """
    Safely loads standard image formats and multi-band 16-bit GeoTIFFs
    into a standardized 8-bit RGB PIL Image for VLM inference.
    """
    image_bytes = file_storage.read()
    filename = (file_storage.filename or "").lower()

    if filename.endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(io.BytesIO(image_bytes)) as src:
                if src.count == 1:
                    raw = src.read(1)
                    norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    return Image.fromarray(norm).convert("RGB")
                else:
                    r = cv2.normalize(src.read(1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    g = cv2.normalize(src.read(2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if src.count >= 2 else r
                    b = cv2.normalize(src.read(3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if src.count >= 3 else r
                    rgb_array = np.stack([r, g, b], axis=-1)
                    return Image.fromarray(rgb_array).convert("RGB")
        except Exception as err:
            print(f"⚠️ Warning: Rasterio GeoTIFF parsing fallback to PIL: {err}")

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------------
# ROUTES FOR THE BLUEPRINT
# ---------------------------------------------------------------------
@vlm_bp.route("/health", methods=["GET"])
def health():
    try:
        import torch
        return jsonify({
            "status": "ok",
            "mode": "remote" if REMOTE_VLM_URL else "local",
            "remote_url": REMOTE_VLM_URL or None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "adapter": ADAPTER_PATH,
            "adapter_exists": Path(ADAPTER_PATH).exists(),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@vlm_bp.route("/predict", methods=["POST"])
def predict():
    # ---- REMOTE MODE: forward to Colab GPU server ----
    if REMOTE_VLM_URL:
        try:
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded."}), 400
            file = request.files["image"]
            prompt = request.form.get("prompt", "").strip()
            if not prompt:
                return jsonify({"error": "Please enter a prompt."}), 400

            resp = requests.post(
                f"{REMOTE_VLM_URL}/predict",
                files={"image": (file.filename, file.stream, file.mimetype)},
                data={"prompt": prompt},
                timeout=120,
            )
            return jsonify(resp.json()), resp.status_code
        except Exception as exc:
            import traceback; traceback.print_exc()
            return jsonify({"success": False, "error": f"Could not reach Colab VLM server: {exc}"}), 500

    # ---- LOCAL MODE: original behavior (needs local GPU) ----
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Please select an image."}), 400

        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Please enter a prompt."}), 400

        image = load_uploaded_image(file)

        with _model_lock:
            clean_text, raw_text = generate_answer(image, prompt)

        boxes = parse_loc_boxes(raw_text, image.width, image.height)
        result_image = draw_boxes(image, boxes) if boxes else image

        return jsonify({
            "success": True,
            "prompt": prompt,
            "answer": clean_text,
            "raw_output": raw_text,
            "boxes": boxes,
            "image": image_to_data_url(result_image),
            "original_width": image.width,
            "original_height": image.height,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500