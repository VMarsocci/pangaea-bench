"""Prediction visualization utilities for the Pangaea evaluation pipeline.

Supports segmentation, regression, and classification tasks.
Saves raw predictions as GeoTIFF and/or visual comparison panels as PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_colormap(num_classes: int) -> np.ndarray:
    """Return an (N, 3) uint8 colour table for *num_classes* class indices."""
    cmap = plt.cm.get_cmap("tab20", max(num_classes, 1))
    lut = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    return lut


def _apply_colormap(
    mask: np.ndarray,
    lut: np.ndarray,
    ignore_index: int = -1,
) -> np.ndarray:
    """Map integer class mask (H, W) → uint8 RGB image using *lut*."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(lut)):
        rgb[mask == cls_id] = lut[cls_id]
    if ignore_index >= 0:
        rgb[mask == ignore_index] = [128, 128, 128]  # grey for ignored pixels
    return rgb


def _extract_rgb(
    image_dict: Dict[str, torch.Tensor],
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
    timestep: int = 0,
) -> np.ndarray:
    """Extract a percentile-stretched uint8 RGB array from an image dict.

    Parameters
    ----------
    image_dict:
        Dict of modality → tensor with shape ``(C, T, H, W)``.
    rgb_bands:
        Indices (0-based within the channel dim) to use as R, G, B channel.
        Falls back to greyscale if bands are out of range.
    timestep:
        Which time step to use (clamped to available range).

    Returns
    -------
    np.ndarray: uint8 array of shape ``(H, W, 3)``.
    """
    # Prefer optical; fall back to first available modality
    key = "optical" if "optical" in image_dict else next(iter(image_dict))
    tensor = image_dict[key].float()  # (C, T, H, W)

    t_idx = min(timestep, tensor.shape[1] - 1)
    img = tensor[:, t_idx].numpy()  # (C, H, W)

    valid = [b for b in rgb_bands if b < img.shape[0]]
    if len(valid) >= 3:
        rgb = img[list(valid[:3])].transpose(1, 2, 0).copy()  # (H, W, 3)
    elif len(valid) > 0:
        grey = img[valid[0]]
        rgb = np.stack([grey, grey, grey], axis=-1)
    else:
        grey = img[0]
        rgb = np.stack([grey, grey, grey], axis=-1)

    # Percentile contrast stretch → [0, 1]
    lo = np.nanpercentile(rgb, 2)
    hi = np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _get_sample_name(metadata: Any, global_idx: int) -> str:
    """Extract a meaningful name from a sample's metadata dict.

    Checks several common keys in priority order, then falls back to a
    zero-padded index string so every name is unique.
    """
    if isinstance(metadata, dict):
        for key in ("filename", "patch_id", "sample_id", "id", "tile_id", "chip_id"):
            val = metadata.get(key)
            if val is not None:
                # Strip extension from filename-like values
                return Path(str(val)).stem
    return f"sample_{global_idx:05d}"


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------


class PredictionVisualizer:
    """Handles saving of raw predictions (GeoTIFF) and visual panels (PNG).

    Parameters
    ----------
    output_dir:
        Root directory for outputs.  Two subdirectories are created:
        ``predictions/`` for raw .tif files and ``visualizations/`` for .png panels.
    save_predictions:
        Save the raw model output as a GeoTIFF (same name as input sample).
    save_visualizations:
        Save a side-by-side comparison panel: input RGB | ground truth | prediction
        (plus an error map for regression).
    max_samples:
        Maximum samples to save. ``-1`` means save every sample.
    rgb_bands:
        Channel indices (0-based) to use for the RGB preview of the input image.
        Default ``(2, 1, 0)`` matches typical Sentinel-2 ordering (R=band3, G=band2,
        B=band1 in 0-indexed).  Override per-evaluator in the task config.
    rank:
        Current distributed rank.  Only rank 0 writes files to avoid duplicates.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_predictions: bool = False,
        save_visualizations: bool = False,
        max_samples: int = -1,
        rgb_bands: Sequence[int] = (2, 1, 0),
        rank: int = 0,
        overwrite: bool = True,
        use_wandb: bool = False,
        vizus_to_wandb: bool = True,
    ) -> None:
        self.save_predictions = save_predictions
        self.save_visualizations = save_visualizations
        self.max_samples = int(max_samples)
        self.rgb_bands = tuple(int(b) for b in rgb_bands)
        self.rank = rank
        self.overwrite = overwrite
        self.use_wandb = use_wandb
        self.vizus_to_wandb = vizus_to_wandb
        self.saved_count = 0
        self.current_tag = "latest"

        self.pred_root: Optional[Path] = None
        self.vis_root: Optional[Path] = None

        if self.is_active:
            output_dir = Path(output_dir)
            if save_predictions:
                self.pred_root = output_dir / "predictions"
                self.pred_root.mkdir(parents=True, exist_ok=True)
            if save_visualizations:
                self.vis_root = output_dir / "visualizations"
                self.vis_root.mkdir(parents=True, exist_ok=True)

    def set_step_tag(self, tag: str) -> None:
        """Set current save tag (e.g. epoch_000, final_model, test_model).

        Resets the saved-sample counter so that each new epoch gets a fresh
        budget of ``max_samples``.  Combined with ``overwrite=True``, this
        ensures only the last epoch's files are kept on disk.
        """
        if not tag:
            self.current_tag = "latest"
        else:
            clean = str(tag).strip().replace(" ", "_")
            self.current_tag = clean or "latest"
        self.saved_count = 0

    def _resolve_output_dir(self, base_dir: Optional[Path]) -> Optional[Path]:
        if base_dir is None:
            return None
        if self.overwrite:
            return base_dir
        tagged = base_dir / self.current_tag
        tagged.mkdir(parents=True, exist_ok=True)
        return tagged

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if any output is expected and this is rank 0."""
        return (self.save_predictions or self.save_visualizations) and self.rank == 0

    def should_save(self) -> bool:
        """Return True if the budget for samples has not been exhausted."""
        if not self.is_active:
            return False
        return self.max_samples < 0 or self.saved_count < self.max_samples

    # ------------------------------------------------------------------
    # Task-specific save methods
    # ------------------------------------------------------------------

    def save_seg_sample(
        self,
        sample_name: str,
        image_dict: Dict[str, torch.Tensor],
        target: torch.Tensor,
        pred: torch.Tensor,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -1,
    ) -> None:
        """Save one segmentation sample.

        Parameters
        ----------
        sample_name:
            Base filename (without extension) derived from dataset metadata.
        image_dict:
            Single-sample image dict, values shaped ``(C, T, H, W)``.
        target:
            Ground-truth integer mask ``(H, W)``.
        pred:
            Predicted integer class mask ``(H, W)``.
        num_classes, class_names, ignore_index:
            Dataset class information used for colormap and legend.
        """
        if not self.is_active:
            return

        pred_np = pred.cpu().numpy().astype(np.int16)
        target_np = target.cpu().numpy().astype(np.int16)

        pred_dir = self._resolve_output_dir(self.pred_root)
        vis_dir = self._resolve_output_dir(self.vis_root)

        if self.save_predictions and pred_dir is not None:
            tifffile.imwrite(str(pred_dir / f"{sample_name}.tif"), pred_np)

        if self.save_visualizations:
            lut = _build_colormap(num_classes)
            rgb_input = _extract_rgb(image_dict, self.rgb_bands)
            gt_rgb = _apply_colormap(target_np, lut, ignore_index)
            pred_rgb = _apply_colormap(pred_np, lut, ignore_index)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb_input)
            axes[0].set_title("Input (RGB)")
            axes[1].imshow(gt_rgb)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_rgb)
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis("off")

            if class_names:
                labels = class_names[:num_classes]
                handles = [
                    plt.Rectangle((0, 0), 1, 1, color=np.array(lut[i]) / 255.0)
                    for i in range(min(num_classes, len(lut)))
                ]
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=min(6, len(labels)),
                    bbox_to_anchor=(0.5, -0.05),
                    fontsize=7,
                )

            fig.tight_layout()
            if vis_dir is not None:
                fig.savefig(
                    str(vis_dir / f"{sample_name}.png"), dpi=100, bbox_inches="tight"
                )
            if self.use_wandb and self.vizus_to_wandb:
                import wandb

                wandb.log(
                    {f"seg_predictions/{sample_name}": wandb.Image(fig)}, commit=False
                )
            plt.close(fig)

        self.saved_count += 1

    def save_reg_sample(
        self,
        sample_name: str,
        image_dict: Dict[str, torch.Tensor],
        target: torch.Tensor,
        pred: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Save one regression sample.

        For multi-channel regression (e.g. FAPAR + FCOVER + LAI), each target
        variable is plotted in its own column triplet: GT | Pred | |Error|.

        Parameters
        ----------
        sample_name:
            Base filename (without extension).
        image_dict:
            Single-sample image dict, values shaped ``(C, T, H, W)``.
        target:
            Ground-truth float array of shape ``(H, W)`` or ``(C, H, W)``.
        pred:
            Predicted float array with the same shape.
        class_names:
            Human-readable name for each regression target channel.
        """
        if not self.is_active:
            return

        pred_np = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()

        pred_dir = self._resolve_output_dir(self.pred_root)
        vis_dir = self._resolve_output_dir(self.vis_root)

        if self.save_predictions and pred_dir is not None:
            tifffile.imwrite(str(pred_dir / f"{sample_name}.tif"), pred_np)

        if self.save_visualizations:
            # Normalise to 3-D (C, H, W) even for single-channel outputs
            if pred_np.ndim == 2:
                pred_np = pred_np[np.newaxis]
                target_np = target_np[np.newaxis]

            n_targets = pred_np.shape[0]
            names = (
                list(class_names[:n_targets])
                if class_names
                else [f"target_{i}" for i in range(n_targets)]
            )

            rgb_input = _extract_rgb(image_dict, self.rgb_bands)
            ncols = 1 + 3 * n_targets  # [input] + [GT, Pred, Error] × n_targets
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            if ncols == 1:
                axes = [axes]

            axes[0].imshow(rgb_input)
            axes[0].set_title("Input (RGB)")
            axes[0].axis("off")

            col = 1
            for gt, pr, nm in zip(target_np, pred_np, names):
                vmin = float(np.nanmin([gt, pr]))
                vmax = float(np.nanmax([gt, pr]))
                err = np.abs(pr - gt)

                im = axes[col].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[col].set_title(f"GT: {nm}")
                axes[col].axis("off")
                fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

                im = axes[col + 1].imshow(pr, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[col + 1].set_title(f"Pred: {nm}")
                axes[col + 1].axis("off")
                fig.colorbar(im, ax=axes[col + 1], fraction=0.046, pad=0.04)

                im = axes[col + 2].imshow(err, cmap="Reds")
                axes[col + 2].set_title(f"|Error|: {nm}")
                axes[col + 2].axis("off")
                fig.colorbar(im, ax=axes[col + 2], fraction=0.046, pad=0.04)

                col += 3

            fig.tight_layout()
            if vis_dir is not None:
                fig.savefig(
                    str(vis_dir / f"{sample_name}.png"), dpi=100, bbox_inches="tight"
                )
            if self.use_wandb and self.vizus_to_wandb:
                import wandb

                wandb.log(
                    {f"reg_predictions/{sample_name}": wandb.Image(fig)}, commit=False
                )
            plt.close(fig)

        self.saved_count += 1

    def save_cls_sample(
        self,
        sample_name: str,
        image_dict: Dict[str, torch.Tensor],
        target_label: Union[int, str],
        pred_label: Union[int, str],
    ) -> None:
        """Save one classification sample.

        Shows the input RGB image with GT and predicted class as a title.
        (No spatial output to save as GeoTIFF for classification tasks.)

        Parameters
        ----------
        sample_name:
            Base filename (without extension).
        image_dict:
            Single-sample image dict, values shaped ``(C, T, H, W)``.
        target_label, pred_label:
            Class ID or name for ground truth and prediction.
        """
        if not self.is_active:
            return

        vis_dir = self._resolve_output_dir(self.vis_root)

        if self.save_visualizations:
            rgb_input = _extract_rgb(image_dict, self.rgb_bands)
            correct = target_label == pred_label
            color = "green" if correct else "red"

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(rgb_input)
            ax.set_title(
                f"GT: {target_label}  |  Pred: {pred_label}",
                fontsize=10,
                color=color,
                fontweight="bold",
            )
            ax.axis("off")
            fig.tight_layout()
            if vis_dir is not None:
                fig.savefig(
                    str(vis_dir / f"{sample_name}.png"), dpi=100, bbox_inches="tight"
                )
            if self.use_wandb and self.vizus_to_wandb:
                import wandb

                wandb.log(
                    {f"cls_predictions/{sample_name}": wandb.Image(fig)}, commit=False
                )
            plt.close(fig)

        self.saved_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def log_summary(self) -> None:
        """Log a one-line summary of what was saved."""
        if not self.is_active or self.saved_count == 0:
            return
        parts = []
        pred_dir = self._resolve_output_dir(self.pred_root)
        vis_dir = self._resolve_output_dir(self.vis_root)
        if self.save_predictions and pred_dir:
            parts.append(f"predictions → {pred_dir}")
        if self.save_visualizations and vis_dir:
            parts.append(f"visualizations → {vis_dir}")
        logger.info(f"Saved {self.saved_count} sample(s): {', '.join(parts)}")
