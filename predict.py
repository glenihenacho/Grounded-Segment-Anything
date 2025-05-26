# predict.py
import os
import torch
import numpy as np
from PIL import Image

# Stub imports; replace with actual modules from your repo
from groundingdino.util.inference import load_model as load_dino_model, predict as dino_predict
from segment_anything import SamPredictor, sam_model_registry

# Util
from grounded_sam.util.image import load_image, save_mask

def predict(image: str, text_prompt: str) -> dict:
    print("🔥 predict.py entry reached")
    
    # Load image
    try:
        print("📥 Downloading input image...")
        image_pil, image_np = load_image(image)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        raise

    # Load GroundingDINO
    try:
        print("🔧 Loading GroundingDINO...")
        grounding_model = load_dino_model(
            config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py",
            checkpoint_path="/weights/groundingdino_swinb.pth"
        )
        boxes = dino_predict(
            model=grounding_model,
            image=image_pil,
            text_prompt=text_prompt
        )
        print("✅ GroundingDINO loaded and predicted.")
    except Exception as e:
        print(f"❌ Error loading GroundingDINO: {e}")
        raise

    # Load SAM
    try:
        print("🔧 Loading SAM...")
        sam = sam_model_registry["vit_h"](checkpoint="/weights/sam_vit_h.pth")
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)

        mask, _, _ = predictor.predict(box=boxes[0])
        print("✅ SAM predicted mask.")
    except Exception as e:
        print(f"❌ Error loading SAM or predicting: {e}")
        raise

    # Save mask
    try:
        print("💾 Saving mask image...")
        mask_path = save_mask(mask)
        print(f"✅ Mask saved to {mask_path}")
    except Exception as e:
        print(f"❌ Error saving mask: {e}")
        raise

    return {"mask": mask_path}
