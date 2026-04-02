from pathlib import Path
import json

import cv2
import timm
import torch
from torchvision import transforms


class LetterClassifier:
    def __init__(self, checkpoint_path: str, labels_path: str, image_size: int = 224) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(labels_path, "r", encoding="utf-8") as handle:
            self.labels = json.load(handle)

        state = torch.load(checkpoint_path, map_location=self.device)
        model_name = state.get("model_name", "efficientnet_b3")
        saved_image_size = state.get("image_size", image_size)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=len(self.labels))
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((saved_image_size, saved_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, frame_bgr, bbox):
        if bbox is None:
            return {"label": "", "confidence": 0.0, "top_k": []}

        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {"label": "", "confidence": 0.0, "top_k": []}

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=min(3, len(self.labels)))

        top_k = [
            {"label": self.labels[int(index)], "confidence": float(prob)}
            for prob, index in zip(top_probs.cpu(), top_indices.cpu())
        ]
        return {"label": top_k[0]["label"], "confidence": top_k[0]["confidence"], "top_k": top_k}

    @staticmethod
    def is_ready(checkpoint_path: str, labels_path: str) -> bool:
        return Path(checkpoint_path).exists() and Path(labels_path).exists()
