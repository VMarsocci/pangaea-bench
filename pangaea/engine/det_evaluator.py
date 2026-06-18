import math
import time
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms
from tqdm import tqdm

from pangaea.engine.evaluator import Evaluator

_SCALAR_KEYS = ("map", "map_50", "map_75", "mar_100")


class DetEvaluator(Evaluator):
    """Object detection evaluator backed by torchmetrics MeanAveragePrecision."""

    def __init__(
        self,
        *args,
        iou_type: Literal["bbox", "segm"] = "bbox",
        class_metrics: bool = True,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.iou_type = iou_type
        self.class_metrics = class_metrics
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        t = time.time()
        if model_ckpt_path is not None:
            ckpt = torch.load(
                model_ckpt_path, map_location=self.device, weights_only=False
            )
            model_name = Path(model_ckpt_path).stem
            model.module.load_state_dict(ckpt.get("model", ckpt))
            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        sync_group = (
            torch.distributed.new_group(backend="gloo")
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        # typed as Any: ty constructs a union of abstract Metric.update/compute and
        # MeanAveragePrecision's concrete overrides, then flags the union as invalid.
        metric: Any = MeanAveragePrecision(
            iou_type=self.iou_type,
            class_metrics=self.class_metrics,
            process_group=sync_group,
        )

        for data in tqdm(
            self.val_loader, desc=f"Evaluating {model_name} on {self.split} set"
        ):
            image_dict = {k: v.to(self.device) for k, v in data["image"].items()}
            if self.inference_mode == "sliding":
                preds = self.sliding_detect(model, image_dict)
            else:
                preds = [
                    {
                        "boxes": p["boxes"].cpu(),
                        "scores": p["scores"].cpu(),
                        "labels": p["labels"].cpu(),
                    }
                    for p in model(image_dict)
                ]
            tgts = [
                {"boxes": b.cpu(), "labels": l.cpu()}
                for b, l in zip(data["boxes"], data["labels"])
            ]
            metric.update(preds, tgts)

        raw = metric.compute()
        metrics = {k: float(raw[k].item()) for k in _SCALAR_KEYS if k in raw}
        if "map_per_class" in raw and raw["map_per_class"].numel() > 0:
            metrics["map_per_class"] = raw["map_per_class"].reshape(-1).cpu().tolist()
            # torchmetrics only includes classes that have positive targets, so the
            # `classes` tensor is the authoritative label-index map for `map_per_class`.
            metrics["per_class_labels"] = raw["classes"].reshape(-1).cpu().tolist()
        self.log_metrics(metrics)
        return metrics, time.time() - t

    @torch.no_grad()
    def sliding_detect(self, model, image_dict):
        """Tile each native-resolution image into encoder-input windows, detect per tile,
        map boxes back to full-image coordinates, then merge tiles with class-wise NMS.

        Mirrors SegEvaluator's sliding inference but merges boxes (offset + NMS) instead of
        averaging dense logits, so detection keeps native object scale on both train and eval.
        """
        modality = next(iter(image_dict))
        img = image_dict[modality]
        b, _, _, height, width = img.shape
        input_size = model.module.encoder.input_size
        h = int(math.ceil(height / input_size))
        w = int(math.ceil(width / input_size))
        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        max_batch = self.sliding_inference_batch

        results = []
        for bi in range(b):
            tiles, origins = [], []
            for i in range(h):
                for j in range(w):
                    y0, x0 = int(h_grid[i]), int(w_grid[j])
                    tiles.append(
                        img[
                            bi : bi + 1,
                            :,
                            :,
                            y0 : y0 + input_size,
                            x0 : x0 + input_size,
                        ]
                    )
                    origins.append((x0, y0))
            tiles = torch.cat(tiles, dim=0)
            n_tiles = tiles.shape[0]
            chunk = max_batch if max_batch else n_tiles

            boxes_all, scores_all, labels_all = [], [], []
            for s in range(0, n_tiles, chunk):
                out = model({modality: tiles[s : s + chunk]})
                for k, p in enumerate(out):
                    x0, y0 = origins[s + k]
                    offset = torch.tensor(
                        [x0, y0, x0, y0],
                        device=p["boxes"].device,
                        dtype=p["boxes"].dtype,
                    )
                    boxes_all.append(p["boxes"] + offset)
                    scores_all.append(p["scores"])
                    labels_all.append(p["labels"])

            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            labels = torch.cat(labels_all)
            keep = batched_nms(boxes, scores, labels, self.nms_threshold)
            results.append(
                {
                    "boxes": boxes[keep].cpu(),
                    "scores": scores[keep].cpu(),
                    "labels": labels[keep].cpu(),
                }
            )
        return results

    @torch.no_grad()
    def __call__(self, model, model_name="model", model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def log_metrics(self, metrics: dict) -> None:
        self.logger.info(f"[{self.split}] Detection metrics")
        for key in _SCALAR_KEYS:
            if key in metrics:
                self.logger.info(f"[{self.split}] {key}: {metrics[key]:.4f}")
        per_class = metrics.get("map_per_class", [])
        labels = metrics.get("per_class_labels", [])
        named_aps = [
            (self.classes[lbl] if 0 <= lbl < len(self.classes) else f"class_{lbl}", ap)
            for lbl, ap in zip(labels, per_class)
        ]
        for cls_name, ap in named_aps:
            self.logger.info(
                f"[{self.split}]   {cls_name.ljust(self.max_name_len)}\t{ap:.4f}"
            )

        if self.use_wandb and self.rank == 0:
            payload = {
                f"{self.split}_{k}": v
                for k, v in metrics.items()
                if not isinstance(v, list)
            }
            for cls_name, ap in named_aps:
                payload[f"{self.split}_AP_{cls_name}"] = ap
            wandb.log(payload)
        self.sync_wandb_if_offline()
