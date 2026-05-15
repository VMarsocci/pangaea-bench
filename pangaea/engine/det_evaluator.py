import os
import time
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from pangaea.engine.evaluator import Evaluator

_SCALAR_KEYS = ("map", "map_50", "map_75", "mar_100")


class DetEvaluator(Evaluator):
    """Object detection evaluator backed by torchmetrics MeanAveragePrecision."""

    def __init__(
        self,
        val_loader: DataLoader,
        exp_dir: str | Path,
        device: torch.device,
        inference_mode: str = "whole",
        sliding_inference_batch: int = 0,
        use_wandb: bool = False,
        wandb_offline: bool = False,
        wandb_offline_sync_communication_dir: str | Path = "~/.wandb_osh_communication",
        iou_type: Literal["bbox", "segm"] = "bbox",
        class_metrics: bool = True,
        save_predictions: bool = False,
        save_visualizations: bool = False,
        save_output_dir: str | Path | None = None,
        max_samples_to_visualize: int = -1,
        rgb_bands: list[int] | None = None,
        overwrite: bool = True,
        vizus_to_wandb: bool = True,
    ) -> None:
        super().__init__(
            val_loader=val_loader,
            exp_dir=exp_dir,
            device=device,
            inference_mode=inference_mode,
            sliding_inference_batch=sliding_inference_batch,
            use_wandb=use_wandb,
            wandb_offline=wandb_offline,
            wandb_offline_sync_communication_dir=wandb_offline_sync_communication_dir,
            save_predictions=save_predictions,
            save_visualizations=save_visualizations,
            save_output_dir=save_output_dir,
            max_samples_to_visualize=max_samples_to_visualize,
            rgb_bands=rgb_bands,
            overwrite=overwrite,
            vizus_to_wandb=vizus_to_wandb,
        )
        self.iou_type = iou_type
        self.class_metrics = class_metrics

    @torch.no_grad()
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        t = time.time()
        if model_ckpt_path is not None:
            ckpt = torch.load(
                model_ckpt_path, map_location=self.device, weights_only=False
            )
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
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
