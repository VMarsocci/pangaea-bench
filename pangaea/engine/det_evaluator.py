import time
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torchmetrics.detection import MeanAveragePrecision
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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.iou_type = iou_type
        self.class_metrics = class_metrics

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
