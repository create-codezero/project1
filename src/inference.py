import os
import argparse
import numpy as np
from PIL import Image
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model import SiameseUNetAttention


def load_inference_transforms():
    """
    Load inference transforms with additional_targets for the second Siamese image.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'image_b': 'image'})


@torch.no_grad()
def predict_change_mask(
    model: torch.nn.Module,
    img_a_path: str,
    img_b_path: str,
    device: torch.device,
    threshold: float = 0.5
):
    """
    Inference pipeline for high-resolution satellite image pairs.
    """
    transform = load_inference_transforms()
    
    # Load raw images
    raw_a = Image.open(img_a_path).convert("RGB")
    raw_b = Image.open(img_b_path).convert("RGB")
    orig_w, orig_h = raw_a.size

    img_a_np = np.array(raw_a)
    img_b_np = np.array(raw_b)

    if img_b_np.shape[:2] != img_a_np.shape[:2]:
        img_b_np = cv2.resize(
            img_b_np, (img_a_np.shape[1], img_a_np.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Preprocess both images using Albumentations twin target
    augmented = transform(image=img_a_np, image_b=img_b_np)
    tensor_a = augmented['image'].unsqueeze(0).to(device)
    tensor_b = augmented['image_b'].unsqueeze(0).to(device)

    # Predict
    model.eval()
    logits = model(tensor_a, tensor_b)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize prediction back to original dimensions if altered
    if probs.shape != (orig_h, orig_w):
        probs = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    binary_mask = (probs > threshold).astype(np.uint8) * 255
    return img_b_np, binary_mask, probs


def run_model_inference(model, t1_path, t2_path, output_mask_path, device="cuda", threshold=0.5,
                         probs_output_path=None):
    """
    Wrapper function called directly by the Flask backend app.py.

    CONFIDENCE FIX: this used to throw away the model's per-pixel sigmoid
    probabilities and only keep the thresholded binary mask — that's why
    app.py had no real model confidence to report and fell back to a fake
    area-based formula (bigger blob = higher confidence, regardless of how
    sure the model actually was). Now it also returns the raw probability
    map so app.py can compute genuine per-detection confidence by averaging
    the model's own probabilities inside each detected region.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    _, binary_mask, probs = predict_change_mask(model, t1_path, t2_path, dev, threshold)
    cv2.imwrite(output_mask_path, binary_mask)
    if probs_output_path:
        np.save(probs_output_path, probs.astype(np.float32))
    return output_mask_path, probs


def create_overlay_heatmap(img_b_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Creates a red-highlighted visual overlay of human development over post-change image.
    """
    overlay = img_b_rgb.copy()
    overlay[binary_mask == 255] = [255, 0, 0]  # Highlight changes in Red
    
    # Blend image with overlay
    blended = cv2.addWeighted(img_b_rgb, 0.6, overlay, 0.4, 0)
    return blended


def main():
    parser = argparse.ArgumentParser(description="Inference Script for SIH Change Detection")
    parser.add_argument("--img_a", type=str, required=True, help="Path to image A (Pre-change)")
    parser.add_argument("--img_b", type=str, required=True, help="Path to image B (Post-change)")
    parser.add_argument("--weights", type=str, default="./models/best_siamese_model.pth")
    parser.add_argument("--out_dir", type=str, default="./dashboard/static")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Model Weights
    print("📦 Loading trained model parameters...")
    model = SiameseUNetAttention(pretrained=False).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Handle checkpoint dictionary keys safely
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Run Prediction
    print("🔍 Executing change detection inference...")
    img_b_rgb, mask, probs = predict_change_mask(model, args.img_a, args.img_b, device)

    # Create visualization artifacts
    heatmap = create_overlay_heatmap(img_b_rgb, mask)

    # Save outputs
    mask_path = os.path.join(args.out_dir, "predicted_mask.png")
    overlay_path = os.path.join(args.out_dir, "overlay_heatmap.png")

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    print(f"✅ Prediction Complete!")
    print(f" └─ Binary Mask saved to: {mask_path}")
    print(f" └─ Heatmap Overlay saved to: {overlay_path}")


if __name__ == "__main__":
    main()