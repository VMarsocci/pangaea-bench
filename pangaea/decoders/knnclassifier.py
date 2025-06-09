from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


def knn_predict(
    features_q: Tensor,     # (B, D)
    features_bank: Tensor,  # (N, D)
    labels_bank: Tensor,    # (N,) or (N, C) for multi-label
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
    multi_label: bool = False,
    top_m: int = 5,
) -> Tensor:
    """Run kNN predictions with optional multi-label Top-M strategy."""
    # Compute cos similarity: (B, N)
    sim_matrix = torch.mm(features_q, features_bank.T)
    # Get top-k neighbors
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)  # both (B, K)

    if multi_label:
        # labels_bank: (N, C) one-hot
        B, K = sim_indices.shape
        C = num_classes
        # Gather one-hot neighbor labels → (B, K, C)
        sim_labels = torch.gather(
            labels_bank,
            0,
            sim_indices.view(-1, 1).expand(-1, C)
        ).view(B, K, C)
        # weight by similarity
        weights = (sim_weight / knn_t).exp().unsqueeze(-1)  # (B, K, 1)
        # compute weighted average → (B, C)
        scores = (sim_labels * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        # pick top-M classes per sample
        top_vals, top_idxs = scores.topk(k=top_m, dim=1)        # both (B, M)
        preds = torch.zeros_like(scores)                        # (B, C)
        preds.scatter_(1, top_idxs, 1.0)                        # mark top M as 1
        return preds

    else:
        # Single-label: original logic
        sim_labels = torch.gather(
        labels_bank.expand(features_q.size(0), -1), dim=-1, index=sim_indices)  # (B, K)

        # we do a reweighting of the similarities
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            features_q.size(0) * knn_k, num_classes, device=sim_labels.device
        )  # (B*K, C)

        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )# (B*K, C)

        pred_scores = torch.sum(
            one_hot_label.view(features_q.size(0), -1, num_classes)
            * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )  # (B, C)
        pred_labels = pred_scores.argsort(dim=-1, descending=True)   # (B, C)
        return pred_labels


class KNNClassifier(Decoder):
    """Non-parametric kNN classifier decoder, with optional multi-label Top-M strategy."""
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        knn_k: int,
        knn_t: float,
        topk: Tuple[int, ...] = (1, 5),
        finetune: bool = False,
        normalize: bool = True,
        feature_dtype: torch.dtype | str = torch.float16,
        multi_label: bool = False,
        top_m: int = 5,      # ← threshold parameter
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)
        self.model_name = "knn_probe"
        self.encoder = encoder
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.topk = topk
        self.normalize = normalize
        self.multi_label = multi_label
        self.top_m = top_m

        # Handle dtype argument
        if isinstance(feature_dtype, str):
            feature_dtype = getattr(torch, feature_dtype)
        self.feature_dtype = feature_dtype

        # Feature bank placeholders
        self._bank: Tensor | None = None
        self._bank_labels: Tensor | None = None

        # Freeze encoder parameters
        assert not finetune, "KNN classifier does not support finetuning"
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Dummy parameter so there's at least one trainable tensor
        self.register_parameter("_dummy", torch.nn.Parameter(torch.empty(1)))

    def _extract_features(self, img: dict[str, Tensor]) -> Tensor:
        # Extract features from a batch of images
        if getattr(self.encoder, "multi_temporal", False):
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)
            if getattr(self.encoder, "multi_temporal_output", False):
                feat = [f.squeeze(-3) for f in feat]
        else:
            # remove temporal dimension [B, C, T=1, H, W] → [B, C, H, W]
            imgs = {k: v[:, :, 0, :, :] for k, v in img.items()}
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(imgs)
            else:
                feat = self.encoder(imgs)

        # If encoder returns multi-scale list, up/down-sample to same size then concat
        if isinstance(feat, (list, tuple)):
            target_h = max(f.shape[-2] for f in feat)
            target_w = max(f.shape[-1] for f in feat)
            feat = torch.cat([
                f if f.shape[-2:] == (target_h, target_w)
                else F.interpolate(f, size=(target_h, target_w),
                                   mode="bilinear", align_corners=False)
                for f in feat
            ], dim=1)

        # Global average pool to (B, D)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        return feat.to(self.feature_dtype)

    @torch.no_grad()
    def build_feature_bank(self, train_loader: DataLoader, device):
        feats, labels = [], []
        for batch in tqdm(train_loader, desc="Building kNN bank", leave=True):
            image, target = batch["image"], batch["target"]
            image = {k: v.to(device) for k, v in image.items()}
            target = target.to(device)

            feats.append(self._extract_features(image).cpu())

            if self.multi_label and target.dim() == 1:
                target = F.one_hot(target, num_classes=self.num_classes).float()
            labels.append(target.cpu())

        self._bank = torch.cat(feats).to(device)            # (N, D)
        self._bank_labels = torch.cat(labels).to(device)    # (N,) or (N, C)

    @torch.no_grad()
    def classify(self, img: dict[str, Tensor]) -> Tensor:
        if self._bank is None or self._bank_labels is None:
            raise RuntimeError("Feature bank empty – call build_feature_bank() first")

        q = self._extract_features(img)  # (B, D)
        return knn_predict(
            features_q=q,
            features_bank=self._bank,
            labels_bank=self._bank_labels,
            num_classes=self.num_classes,
            knn_k=self.knn_k,
            knn_t=self.knn_t,
            multi_label=self.multi_label,
            top_m=self.top_m,
        )