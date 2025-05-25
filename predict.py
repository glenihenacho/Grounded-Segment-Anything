# Prediction interface for Cog (Replicate)
import os
import json
from typing import Any
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel

from subprocess import call

# Install GroundingDINO and SAM
HOME = os.getcwd()
os.chdir("GroundingDINO")
call("pip install -q .", shell=True)
os.chdir(HOME)
os.chdir("segment_anything")
call("pip install -q .", shell=True)
os.chdir(HOME)

# Import GroundingDINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Import SAM
from segment_anything import build_sam, build_sam_hq, SamPredictor


class ModelOutput(BaseModel):
    masked_img: Path
    rounding_box_img: Path
    json_data: Any
    tags: str


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_size = 384
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "pretrained/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        self.sam = SamPredictor(
            build_sam(checkpoint="pretrained/sam_vit_h_4b8939.pth").to(self.device)
        )
        self.sam_hq = SamPredictor(
            build_sam_hq(checkpoint="pretrained/sam_hq_vit_h.pth").to(self.device)
        )

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        text_prompt: str = Input(description="Target surface, e.g., 'floor'"),
        use_sam_hq: bool = Input(description="Use SAM HQ for higher quality", default=False),
    ) -> ModelOutput:

        box_threshold = 0.25
        text_threshold = 0.2
        iou_threshold = 0.5

        image_pil, image = load_image(str(input_image))
        raw_image = image_pil.resize((self.image_size, self.image_size))
        raw_image = self.transform(raw_image).unsqueeze(0).to(self.device)

        # Detect bounding boxes with GroundingDINO
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.model, image, text_prompt, box_threshold, text_threshold, device=self.device
        )

        predictor = self.sam_hq if use_sam_hq else self.sam

        image = cv2.imread(str(input_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        # Save binary mask for ControlNet
        combined_mask = (masks.sum(0) > 0).cpu().numpy().astype(np.uint8) * 255
        mask_path = "/tmp/mask_clean.png"
        Image.fromarray(combined_mask).save(mask_path)

        # Save annotated image
        plt.figure(figsize=(10, 10))
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        round_box_path = "/tmp/annotated_boxes.png"
        plt.axis("off")
        plt.savefig(round_box_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()

        json_data = {
            "tags": text_prompt,
            "mask": [{"value": 1, "label": text_prompt}],
            "boxes": [box.tolist() for box in boxes_filt],
        }

        return ModelOutput(
            masked_img=Path(mask_path),
            rounding_box_img=Path(round_box_path),
            json_data=json_data,
            tags=text_prompt,
        )


# Utility functions

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(f"{pred_phrase}({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color \
        else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=1.5))
    ax.text(x0, y0, label)
