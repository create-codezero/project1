# --- FIX FOR NUMPY 2.0+ & TENSORBOARD/ACCELERATE CONFLICT ---
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# -----------------------------------------------------------

import io
import os
import json
import base64
import time
import cv2
import rasterio
from rasterio.enums import Resampling
import torch
import sys
from datetime import datetime
from fpdf import FPDF
from groq import Groq
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from pystac_client import Client
import planetary_computer
from PIL import Image
import urllib.request
import gdown

# Advanced Data Processing Libraries
from skimage.exposure import match_histograms
from shapely.geometry import Polygon, mapping
from scipy.ndimage import uniform_filter

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Add root src directory to python path for Manual mode dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model import SiameseUNetAttention
    from src.inference import run_model_inference
    from src.post_process import process_detected_changes
except ImportError as e:
    import traceback
    print("Warning: Manual mode src modules not found. Ensure src directory is accessible.")
    print(f"❌ ACTUAL ERROR: {e}")
    traceback.print_exc()
    
app = Flask(__name__, template_folder='templates')

# --- DIRECTORIES SETUP ---
SAVE_DIR = os.path.abspath("saved_images")
os.makedirs(SAVE_DIR, exist_ok=True)

UPLOAD_FOLDER = os.path.abspath("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_DIR = os.path.abspath("model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'best_siamese_model.pth')

# --- GOOGLE DRIVE MODEL DOWNLOADER ---
GOOGLE_DRIVE_FILE_ID = "1vaNaT8FkHY-ysYwJWhyoEVOw6A7_vPq-"

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024 * 1024:
        print("📥 Model file not found locally. Downloading from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            print("✅ Model downloaded successfully from Google Drive!")
        except Exception as e:
            print(f"❌ Failed to download model automatically: {e}")

# Run download check on container startup
download_model_from_drive()

# --- STAC CATALOG SETUP (MAP MODE) ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# --- AI MODEL SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

print("🚀 Initializing AI Change Detection Model for Flask...")
try:
    model = SiameseUNetAttention(pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
    else:
        print(f"⚠️ Warning: Model weights not found at {MODEL_PATH}.")
except Exception as e:
    print(f"⚠️ Model Initialization Error: {e}")

# In-memory store for the latest detected events (Shared for PDF, Copilot & GeoJSON)
latest_events = []

# Initialize Groq Client
client = Groq()

# ==========================================
# ADVANCED UTILITY FUNCTIONS
# ==========================================
def calculate_ndvi(nir, red):
    """Calculates Normalized Difference Vegetation Index"""
    denominator = (nir + red)
    denominator[denominator == 0] = 1e-10
    return (nir - red) / denominator

def calculate_rvi(vv, vh):
    """Calculates Radar Vegetation Index (RVI) for Dual-Pol SAR"""
    return (4 * vh) / (vv + vh + 1e-5)

def pixel_to_latlon(x, y, transform):
    lon, lat = transform * (x, y)
    return float(lat), float(lon)

def crop_to_base64(img_array, bbox_crop):
    x, y, w, h = bbox_crop
    pad = 20
    h_img, w_img = img_array.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

    cropped = img_array[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def contour_to_geojson_polygon(contour, transform):
    points = []
    for point in contour:
        x, y = point[0]
        lon, lat = pixel_to_latlon(x, y, transform)
        points.append((lon, lat))
    
    if len(points) >= 3:
        points.append(points[0])
        return mapping(Polygon(points))
    return None

# ==========================================
# OPTICAL PIPELINE (SENTINEL-2)
# ==========================================
def fetch_optimized_imagery(bbox, date_range, label="image"):
    print(f"\n--- Searching STAC Catalog (Optical) ---")
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}}, sortby=["eo:cloud_cover"]
    )
    
    items = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-2 imagery found for date range: {date_range}")

    selected_item = items[0]
    red_url = selected_item.assets["B04"].href
    green_url = selected_item.assets["B03"].href
    blue_url = selected_item.assets["B02"].href
    nir_url = selected_item.assets["B08"].href
    scl_url = selected_item.assets["SCL"].href

    min_lon, min_lat, max_lon, max_lat = bbox

    with rasterio.open(red_url) as src:
        native_crs = src.crs
        native_bounds = transform_bounds("EPSG:4326", native_crs, min_lon, min_lat, max_lon, max_lat)
        window = window_from_bounds(*native_bounds, transform=src.transform)
        
        with rasterio.open(red_url) as r_src, rasterio.open(green_url) as g_src, rasterio.open(blue_url) as b_src, rasterio.open(nir_url) as nir_src:
            red = r_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            green = g_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            blue = b_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            nir = nir_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            out_shape = red.shape

        with rasterio.open(scl_url) as scl_src:
            scl_window = window_from_bounds(*native_bounds, transform=scl_src.transform)
            scl = scl_src.read(1, window=scl_window, out_shape=out_shape, resampling=Resampling.nearest)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    
    ndvi = calculate_ndvi(nir, red)
    rgb = np.clip((rgb / 3000.0) * 255, 0, 255).astype(np.uint8)
    
    height, width, _ = rgb.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return rgb, ndvi, scl, transform

# ==========================================
# SAR PIPELINE (SENTINEL-1)
# ==========================================
def lee_filter(img, size=3):
    """Applies a mathematical Lee filter for SAR speckle reduction."""
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-5)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def fetch_optimized_sar(bbox, date_range, label="sar_image"):
    print(f"\n--- Searching STAC Catalog (SAR) ---")
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=date_range)
    
    items = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-1 SAR imagery found for date range: {date_range}")

    selected_item = items[0]
    vv_url = selected_item.assets["vv"].href
    vh_url = selected_item.assets["vh"].href
    min_lon, min_lat, max_lon, max_lat = bbox

    with rasterio.open(vv_url) as src:
        native_crs = src.crs
        native_bounds = transform_bounds("EPSG:4326", native_crs, min_lon, min_lat, max_lon, max_lat)
        window = window_from_bounds(*native_bounds, transform=src.transform)
        
        with rasterio.open(vh_url) as vh_src:
            vv = src.read(1, window=window, boundless=True, fill_value=1e-5).astype(np.float32)
            vh = vh_src.read(1, window=window, boundless=True, fill_value=1e-5).astype(np.float32)

    vv = np.clip(vv, 1e-5, None)
    vh = np.clip(vh, 1e-5, None)
    
    vv_filtered = lee_filter(vv, size=5)
    vh_filtered = lee_filter(vh, size=5)

    vv_db = 10 * np.log10(vv_filtered)
    vh_db = 10 * np.log10(vh_filtered)

    def normalize_db(db_img, min_db, max_db):
        norm = (db_img - min_db) / (max_db - min_db)
        return np.clip(norm * 255, 0, 255).astype(np.uint8)

    r = normalize_db(vv_db, -25, 0)
    g = normalize_db(vh_db, -30, -5)
    ratio = np.clip(vv_filtered / (vh_filtered + 1e-5), 1e-5, None)
    ratio_db = 10 * np.log10(ratio)
    b = normalize_db(ratio_db, 0, 15)

    pseudo_rgb = np.stack([r, g, b], axis=-1)
    height, width = vv.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return pseudo_rgb, vv_filtered, vh_filtered, transform

def generate_sar_change_mask(t1_vv, t1_vh, t2_vv, t2_vh):
    """Generates an independent SAR change mask using Log-Ratio and Otsu thresholding."""
    ratio_vv = np.abs(10 * np.log10(t2_vv / t1_vv))
    ratio_vh = np.abs(10 * np.log10(t2_vh / t1_vh))
    combined_change = (ratio_vv + ratio_vh) / 2.0
    
    change_norm = cv2.normalize(combined_change, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(change_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/saved_images/<path:filename>')
def serve_saved_images(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/estimate', methods=['POST'])
def estimate_time():
    data = request.json
    bbox = data.get('bbox')
    modality = data.get('modality', 'optical').lower()
    try:
        min_lon, min_lat, max_lon, max_lat = bbox
        area_km2 = abs((max_lat - min_lat) * 111 * (max_lon - min_lon) * 111 * np.cos((min_lat + max_lat) / 2 * np.pi / 180))
        
        # Fusion takes the longest since it fetches 4 full scenes and runs dual AI pipelines
        base_time = 30 if modality == 'fusion' else (20 if modality == 'sar' else 15)
        est_seconds = min(base_time + int(area_km2 / 5), 180)
        return jsonify({"estimated_seconds": est_seconds})
    except Exception as e:
        return jsonify({"estimated_seconds": 45})

@app.route('/api/detect-satellite', methods=['POST'])
def detect_satellite():
    global latest_events
    data = request.json
    if not data: return jsonify({"status": "error", "error": "No payload"}), 400

    bbox = data.get('bbox')
    t1_date = data.get('t1_date')
    t2_date = data.get('t2_date')
    modality = data.get('modality', 'optical').lower().strip() # Ensures no stray spaces

    events = []
    event_id = 1
    timestamp = int(time.time())
    sat_mask_path = os.path.join(SAVE_DIR, f"sat_mask_{timestamp}.png")
    t1_full_name = f"annotated_t1_{timestamp}.png"
    t2_full_name = f"annotated_t2_{timestamp}.png"

    try:
        # ---------------------------------------------------------
        # 1. OPTICAL PIPELINE
        # ---------------------------------------------------------
        if modality == 'optical':
            if model is None: return jsonify({"status": "error", "error": "AI Model not loaded."}), 500
            t1_rgb, t1_ndvi, t1_scl, transform = fetch_optimized_imagery(bbox, t1_date)
            t2_rgb, t2_ndvi, t2_scl, _ = fetch_optimized_imagery(bbox, t2_date)

            if t1_rgb.shape != t2_rgb.shape:
                t2_rgb = cv2.resize(t2_rgb, (t1_rgb.shape[1], t1_rgb.shape[0]))
                t2_ndvi = cv2.resize(t2_ndvi, (t1_rgb.shape[1], t1_rgb.shape[0]))
                t2_scl = cv2.resize(t2_scl, (t1_rgb.shape[1], t1_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            t2_rgb = match_histograms(t2_rgb, t1_rgb, channel_axis=-1)
            sat_t1_path = os.path.join(SAVE_DIR, f"t1_{timestamp}.png")
            sat_t2_path = os.path.join(SAVE_DIR, f"t2_{timestamp}.png")
            Image.fromarray(t1_rgb).save(sat_t1_path)
            Image.fromarray(t2_rgb).save(sat_t2_path)

            run_model_inference(model, sat_t1_path, sat_t2_path, sat_mask_path, device=DEVICE)
            mask_img = cv2.imread(sat_mask_path, cv2.IMREAD_GRAYSCALE)
            
            cloud_mask = np.isin(t1_scl, [3, 8, 9, 10]) | np.isin(t2_scl, [3, 8, 9, 10])
            mask_img[cloud_mask] = 0

            _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            t1_annotated, t2_annotated = t1_rgb.copy(), t2_rgb.copy()

            for cnt in contours:
                area_pixels = cv2.contourArea(cnt)
                if area_pixels < 25: continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(t1_annotated, (x, y), (x + w, y + h), (255, 0, 0), 3) # Blue
                cv2.rectangle(t2_annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                lat, lon = pixel_to_latlon(x + w // 2, y + h // 2, transform)
                
                c_mask = np.zeros(mask_img.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                
                mean_ndvi_t1 = np.mean(t1_ndvi[c_mask == 255])
                mean_ndvi_t2 = np.mean(t2_ndvi[c_mask == 255])
                area_sq_m = int(area_pixels * 100)
                
                activity = "Large Infrastructure Construction" if area_sq_m > 5000 else "Minor Change"
                if mean_ndvi_t1 > 0.3 and mean_ndvi_t2 < 0.15: activity = "Deforestation"
                elif mean_ndvi_t1 < 0.1 and mean_ndvi_t2 > 0.3: activity = "Afforestation"

                events.append({
                    "id": event_id, "lat": lat, "lon": lon, "activity_type": activity,
                    "severity": "HIGH" if area_sq_m > 5000 else "MEDIUM",
                    "confidence": round(float(np.min([98.5, 65.0 + (area_pixels / 10.0)])), 1),
                    "area_sq_m": area_sq_m,
                    "t1_patch": crop_to_base64(t1_rgb, [x, y, w, h]), "t2_patch": crop_to_base64(t2_rgb, [x, y, w, h]),
                    "geometry": contour_to_geojson_polygon(cnt, transform)
                })
                event_id += 1

        # ---------------------------------------------------------
        # 2. SAR PIPELINE
        # ---------------------------------------------------------
        elif modality == 'sar':
            t1_pseudo, t1_vv, t1_vh, transform = fetch_optimized_sar(bbox, t1_date)
            t2_pseudo, t2_vv, t2_vh, _ = fetch_optimized_sar(bbox, t2_date)

            if t1_vv.shape != t2_vv.shape:
                t2_vv = cv2.resize(t2_vv, (t1_vv.shape[1], t1_vv.shape[0]))
                t2_vh = cv2.resize(t2_vh, (t1_vh.shape[1], t1_vh.shape[0]))
                t2_pseudo = cv2.resize(t2_pseudo, (t1_pseudo.shape[1], t1_pseudo.shape[0]))

            mask_img = generate_sar_change_mask(t1_vv, t1_vh, t2_vv, t2_vh)
            Image.fromarray(mask_img).save(sat_mask_path)

            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            t1_annotated, t2_annotated = t1_pseudo.copy(), t2_pseudo.copy()

            for cnt in contours:
                area_pixels = cv2.contourArea(cnt)
                if area_pixels < 25: continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(t1_annotated, (x, y), (x + w, y + h), (0, 255, 255), 3) # Yellow
                cv2.rectangle(t2_annotated, (x, y), (x + w, y + h), (0, 255, 255), 3)

                lat, lon = pixel_to_latlon(x + w // 2, y + h // 2, transform)
                
                c_mask = np.zeros(mask_img.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                
                mean_t1_vv = np.mean(t1_vv[c_mask == 255])
                mean_t2_vv = np.mean(t2_vv[c_mask == 255])
                area_sq_m = int(area_pixels * 100)

                if mean_t2_vv > mean_t1_vv * 1.3: activity = "New Structure (SAR)"
                elif mean_t1_vv > mean_t2_vv * 1.3: activity = "Land Clearing (SAR)"
                else: activity = "Surface Roughness Change (SAR)"

                events.append({
                    "id": event_id, "lat": lat, "lon": lon, "activity_type": activity,
                    "severity": "HIGH" if area_sq_m > 5000 else "MEDIUM",
                    "confidence": round(float(np.min([96.0, 70.0 + (area_pixels / 8.0)])), 1),
                    "area_sq_m": area_sq_m,
                    "t1_patch": crop_to_base64(t1_pseudo, [x, y, w, h]), "t2_patch": crop_to_base64(t2_pseudo, [x, y, w, h]),
                    "geometry": contour_to_geojson_polygon(cnt, transform)
                })
                event_id += 1

        # ---------------------------------------------------------
        # 3. MULTI-MODAL FUSION PIPELINE (ADVANCED)
        # ---------------------------------------------------------
        elif modality == 'fusion':
            if model is None: return jsonify({"status": "error", "error": "AI Model required for fusion."}), 500
            
            # Fetch Optical (Master Grid)
            t1_rgb, t1_ndvi, t1_scl, transform = fetch_optimized_imagery(bbox, t1_date)
            t2_rgb, t2_ndvi, t2_scl, _ = fetch_optimized_imagery(bbox, t2_date)
            
            # Fetch SAR 
            _, t1_vv, t1_vh, _ = fetch_optimized_sar(bbox, t1_date)
            _, t2_vv, t2_vh, _ = fetch_optimized_sar(bbox, t2_date)

            # Coregistration: Snap Optical and SAR to exactly the same matrix space
            target_shape = (t1_rgb.shape[1], t1_rgb.shape[0]) # (width, height)
            t2_rgb = cv2.resize(t2_rgb, target_shape)
            t2_ndvi = cv2.resize(t2_ndvi, target_shape)
            t2_scl = cv2.resize(t2_scl, target_shape, interpolation=cv2.INTER_NEAREST)
            
            t1_vv = cv2.resize(t1_vv, target_shape)
            t1_vh = cv2.resize(t1_vh, target_shape)
            t2_vv = cv2.resize(t2_vv, target_shape)
            t2_vh = cv2.resize(t2_vh, target_shape)

            # Generate Optical Mask
            t2_rgb = match_histograms(t2_rgb, t1_rgb, channel_axis=-1)
            sat_t1_path = os.path.join(SAVE_DIR, f"opt_t1_{timestamp}.png")
            sat_t2_path = os.path.join(SAVE_DIR, f"opt_t2_{timestamp}.png")
            Image.fromarray(t1_rgb).save(sat_t1_path)
            Image.fromarray(t2_rgb).save(sat_t2_path)
            run_model_inference(model, sat_t1_path, sat_t2_path, sat_mask_path, device=DEVICE)
            
            opt_mask = cv2.imread(sat_mask_path, cv2.IMREAD_GRAYSCALE)
            cloud_mask = np.isin(t1_scl, [3, 8, 9, 10]) | np.isin(t2_scl, [3, 8, 9, 10])
            opt_mask[cloud_mask] = 0
            _, opt_thresh = cv2.threshold(opt_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Generate SAR Mask
            sar_mask = generate_sar_change_mask(t1_vv, t1_vh, t2_vv, t2_vh)

            # FUSION LOGIC: Bitwise OR to capture all events
            fusion_mask = cv2.bitwise_or(opt_thresh, sar_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            fusion_mask = cv2.morphologyEx(fusion_mask, cv2.MORPH_OPEN, kernel)
            
            Image.fromarray(fusion_mask).save(sat_mask_path)
            contours, _ = cv2.findContours(fusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Base visualization on Optical as it is most readable for users
            t1_annotated, t2_annotated = t1_rgb.copy(), t2_rgb.copy()

            for cnt in contours:
                area_pixels = cv2.contourArea(cnt)
                if area_pixels < 25: continue
                
                c_mask = np.zeros(fusion_mask.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)

                # Cross-Verification Metrics
                opt_overlap = np.count_nonzero(cv2.bitwise_and(opt_thresh, c_mask)) / area_pixels
                sar_overlap = np.count_nonzero(cv2.bitwise_and(sar_mask, c_mask)) / area_pixels
                under_cloud = np.count_nonzero(cloud_mask[c_mask == 255]) / area_pixels > 0.5

                # Semantic Data Extraction
                mean_ndvi_t1 = np.mean(t1_ndvi[c_mask == 255])
                mean_ndvi_t2 = np.mean(t2_ndvi[c_mask == 255])
                
                # Radar Vegetation Index (RVI) calculation
                rvi_t1 = calculate_rvi(t1_vv, t1_vh)
                rvi_t2 = calculate_rvi(t2_vv, t2_vh)
                mean_rvi_t1 = np.mean(rvi_t1[c_mask == 255])
                mean_rvi_t2 = np.mean(rvi_t2[c_mask == 255])
                mean_vv_t1 = np.mean(t1_vv[c_mask == 255])
                mean_vv_t2 = np.mean(t2_vv[c_mask == 255])

                x, y, w, h = cv2.boundingRect(cnt)
                lat, lon = pixel_to_latlon(x + w // 2, y + h // 2, transform)
                area_sq_m = int(area_pixels * 100)

                # Advanced Confidence Routing
                base_conf = 70.0 + (area_pixels / 10.0)
                if opt_overlap > 0.3 and sar_overlap > 0.3:
                    conf = base_conf + 18.0  # High sensor agreement
                    box_color = (255, 0, 255) # Purple for Fused
                elif sar_overlap > 0.3 and under_cloud:
                    conf = base_conf + 12.0  # SAR penetrated cloud
                    box_color = (0, 255, 255) # Yellow for SAR
                elif opt_overlap > 0.3:
                    conf = base_conf  # Optical only, clear skies
                    box_color = (255, 0, 0) # Red for Optical
                elif sar_overlap > 0.3:
                    conf = base_conf - 5.0 # SAR anomaly without visual confirmation
                    box_color = (0, 255, 255) # Yellow for SAR
                else:
                    continue # Discard ghost artifacts

                conf = round(float(np.clip(conf, 0.0, 99.9)), 1)

                # Advanced Multi-Modal Classification Matrix
                activity = "Unclassified Terrain Change"
                severity = "LOW"
                
                if mean_ndvi_t1 > 0.3 and mean_ndvi_t2 < 0.15:
                    if mean_vv_t2 > mean_vv_t1 * 1.2:
                        activity = "Vegetation Cleared -> New Structures (Verified)"
                        severity = "HIGH"
                    else:
                        activity = "Deforestation / Canopy Loss"
                        severity = "HIGH"
                elif mean_vv_t2 > mean_vv_t1 * 1.3 and mean_rvi_t2 < mean_rvi_t1 * 0.8:
                    activity = "Construction / Volumetric Assembly"
                    severity = "HIGH" if area_sq_m > 3000 else "MEDIUM"
                elif under_cloud and sar_overlap > 0.3:
                    activity = "SAR Cloud-Penetration Detection (Likely Structural)"
                    severity = "MEDIUM"

                # Draw dynamically colored bounding boxes
                cv2.rectangle(t1_annotated, (x, y), (x + w, y + h), box_color, 3)
                cv2.rectangle(t2_annotated, (x, y), (x + w, y + h), box_color, 3)

                events.append({
                    "id": event_id, "lat": lat, "lon": lon, "activity_type": activity,
                    "severity": severity, "confidence": conf, "area_sq_m": area_sq_m,
                    "t1_patch": crop_to_base64(t1_rgb, [x, y, w, h]), 
                    "t2_patch": crop_to_base64(t2_rgb, [x, y, w, h]),
                    "geometry": contour_to_geojson_polygon(cnt, transform)
                })
                event_id += 1
                
        else:
            return jsonify({"status": "error", "error": f"Invalid modality: {modality}"}), 400

        # Save annotated Full Output
        Image.fromarray(t1_annotated).save(os.path.join(SAVE_DIR, t1_full_name))
        Image.fromarray(t2_annotated).save(os.path.join(SAVE_DIR, t2_full_name))
        
        latest_events = events
        
        return jsonify({
            "status": "success", "events_count": len(events), "events": events,
            "t1_full_url": f"/saved_images/{t1_full_name}", "t2_full_url": f"/saved_images/{t2_full_name}",
            "mask_url": f"/saved_images/sat_mask_{timestamp}.png"
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect_changes():
    global latest_events
    if 'pre_image' not in request.files or 'post_image' not in request.files:
        return jsonify({"error": "Please provide both pre and post change images."}), 400
        
    t1_file = request.files['pre_image']
    t2_file = request.files['post_image']
    t1_path = os.path.join(app.config['UPLOAD_FOLDER'], 't1.png')
    t2_path = os.path.join(app.config['UPLOAD_FOLDER'], 't2.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_mask.png')
    
    t1_file.save(t1_path)
    t2_file.save(t2_path)
    run_model_inference(model, t1_path, t2_path, mask_path, device=DEVICE)
    
    events, annotated_path = process_detected_changes(
        mask_path=mask_path, post_image_path=t2_path, output_dir=app.config['UPLOAD_FOLDER']
    )
    
    latest_events = events
    timestamp = int(time.time())
    
    return jsonify({
        "status": "success", "events_count": len(events),
        "annotated_image": f"/uploads/{os.path.basename(annotated_path)}?t={timestamp}",
        "mask_image": f"/uploads/predicted_mask.png?t={timestamp}",
        "events": events
    })

@app.route('/api/download-geojson', methods=['GET'])
def download_geojson():
    global latest_events
    features = []
    for event in latest_events:
        if event.get('geometry'):
            features.append({
                "type": "Feature",
                "properties": {
                    "id": event.get("id"), "activity_type": event.get("activity_type"),
                    "severity": event.get("severity"), "confidence": event.get("confidence"),
                    "area_sq_m": event.get("area_sq_m")
                },
                "geometry": event.get("geometry")
            })
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/download-pdf', methods=['GET'])
def download_pdf():
    global latest_events
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 8, 'Executive Summary', 0, 1)
    
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 6, 
        f"Total Violations / Events Detected: {len(latest_events)}\n"
        "This document summarizes the semantic change classification results "
        "and spatial expansion metrics automatically flagged by the monitoring pipeline."
    )
    pdf.ln(5)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Detected Violation Events', 0, 1)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(220, 224, 230)
    pdf.cell(15, 8, 'ID', 1, 0, 'CENTER', True)
    pdf.cell(75, 8, 'Activity Type (Semantic AI)', 1, 0, 'LEFT', True)
    pdf.cell(35, 8, 'Area (m²)', 1, 0, 'CENTER', True)
    pdf.cell(30, 8, 'Confidence', 1, 0, 'CENTER', True)
    pdf.cell(35, 8, 'Severity', 1, 1, 'CENTER', True)
    
    pdf.set_font('helvetica', '', 9)
    for event in latest_events:
        event_id = str(event.get('id', event.get('ID', '-')))
        event_type = str(event.get('activity_type', event.get('Type', event.get('Activity Type', '-'))))
        event_area = str(event.get('area_sq_m', event.get('Area', '-')))
        event_conf = str(event.get('confidence', event.get('Confidence', '-')))
        event_sev = str(event.get('severity', event.get('Severity', '-')))

        pdf.cell(15, 8, event_id, 1, 0, 'CENTER')
        pdf.cell(75, 8, event_type, 1, 0, 'LEFT')
        pdf.cell(35, 8, event_area, 1, 0, 'CENTER')
        pdf.cell(30, 8, event_conf, 1, 0, 'CENTER')
        pdf.cell(35, 8, event_sev, 1, 1, 'CENTER')
        
    pdf_bytes = pdf.output()
    pdf_buffer = io.BytesIO(bytes(pdf_bytes))
    
    return send_file(
        pdf_buffer, as_attachment=True,
        download_name=f"change_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mimetype='application/pdf'
    )

@app.route('/api/copilot', methods=['POST'])
def copilot_query():
    global latest_events
    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not latest_events:
        return jsonify({"response": "⚠️ Please run a change detection pipeline (Map or Manual) first!"})

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"response": "⚠️ GROQ_API_KEY is not set in your environment variables."})

    context_str = f"Current Change Detection & Violation Events Summary ({len(latest_events)} events detected):\n"
    for e in latest_events:
        context_str += (
            f"- ID: #{e.get('id', '-')}, Activity: {e.get('activity_type', '-')}, "
            f"Area: {e.get('area_sq_m', '-')} m², Confidence: {e.get('confidence', '-')}%.\n"
        )

    system_prompt = (
        "You are an expert Geospatial AI Compliance Copilot. "
        "Assist users in analyzing satellite change detection and human activity violation events. "
        "Answer the user's question clearly and concisely based strictly on the provided data context below. "
        "IMPORTANT: If the user asks to export, download, or generate a PDF report, provide exactly this markdown link: <a href='/api/download-pdf' target='_blank'>📥 Download PDF Report</a>\n"
        "If the user asks to export or download GeoJSON, provide exactly this markdown link: <a href='/api/download-geojson' target='_blank'>🗺️ Download GeoJSON</a>\n\n"
        f"Data Context:\n{context_str}"
    )

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3, max_tokens=600,
        )
        return jsonify({"response": completion.choices[0].message.content})

    except Exception as e:
        return jsonify({"response": f"❌ Error connecting to Groq API: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)