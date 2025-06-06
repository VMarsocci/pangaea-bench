import torch
from typing import Tuple
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


# Obtained from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking/knn.py
def knn_predict(
    features_q: Tensor,  # (B, D)
    features_bank: Tensor,  # (N, D)
    labels_bank: Tensor,  # (N,)
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
    ) -> Tensor:
    """Run kNN predictions on features based on a feature bank .Non-parametric prediction (InstDisc / MoCo style). 

    This method is commonly used to monitor the performance of self-supervised
    learning methods. The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor of shape (B, D) for which you want predictions, where B is the
            batch size and D is the feature dimension.
        feature_bank:
            Tensor of shape (N, D) representing a database of features used for kNN,
            where N is the number of stored feature vectors.
        feature_labels:
            Tensor of shape (N,) containing labels for the corresponding
            feature vectors in the feature_bank.
        num_classes:
            Number of classes (e.g., `10` for CIFAR-10).
        knn_k:
            Number of k nearest neighbors used for kNN.
        knn_t:
            Temperature parameter to reweight similarities for kNN.

    Returns:
        Tensor of shape (B, num_classes) with the predicted class indices sorted
        by probability in descending order for each sample. The first index
        corresponds to the most probable class. To get the top-1 prediction,
        you can access `pred_labels[:, 0]`.
    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(features_q, features_bank.T)
    
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)  # (B, K)
    
    sim_labels = torch.gather(
        labels_bank.expand(features_q.size(0), -1), dim=-1, index=sim_indices
    )  # (B, K)
    
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
    """Non-parametric decoder – holds only the frozen encoder + feature bank.""" 
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
    ):
        """ KNN classifier decoder.
        Args:
            encoder (Encoder): Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            num_classes (int): number of classes in the dataset.
            knn_k (int): number of neighbours used for KNN search.
            knn_t (float): temperature parameter to reweight similarities.
            topk (int): Tuple of integers defining the top-k accuracy metrics to compute.
            finetune (bool): whether to finetune the encoder.
        """
        
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )
        
        self.model_name = "knn_probe"
        self.encoder = encoder
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.topk = topk
        self.normalize = normalize
        if isinstance(feature_dtype, str):
            try:
                feature_dtype = getattr(torch, feature_dtype)
            except AttributeError:
                raise ValueError(
                    f"Unknown dtype string '{feature_dtype}'. "
                    "Use a torch.dtype or e.g. 'float16', 'bfloat16', 'float32'."
                )
        self.feature_dtype = feature_dtype
        self._bank: Tensor | None = None  # (N, D)
        self._bank_labels: Tensor | None = None

        assert self.finetune == False, "KNN classifier does not support finetuning"
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Now model.parameters() contains at least one trainable tensor, so DDP is happy. 
        self.register_parameter("_dummy", torch.nn.Parameter(torch.empty(1)))  # A dummy parameter to keep .parameters() non-empty

    
    def _extract_features(self, img: dict[str, Tensor]) -> Tensor:
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)
            # multi_temporal models can return either (B C' T>1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]
        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
    
        # Resize multi-scale outputs → same (H, W) and concat along channels
        if isinstance(feat, (list, tuple)):
            target_h = max(f.shape[-2] for f in feat)
            target_w = max(f.shape[-1] for f in feat)
            feat = torch.cat(
                [
                    f
                    if f.shape[-2:] == (target_h, target_w)
                    else F.interpolate(f, size=(target_h, target_w),
                                       mode="bilinear", align_corners=False)
                    for f in feat
                ],
                dim=1,
            )  # (B, ΣC, H, W)
        
        # Global-avg-pool → (B, D)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        return feat.to(self.feature_dtype)  # (B, D)
    
    @torch.no_grad()
    def build_feature_bank(self, train_loader, device):
        
        feats, labels = [], []
        for batch in tqdm(train_loader,
                          desc=f"Building kNN bank)", leave=False):
            image, target = batch["image"], batch["target"]
            image = {k: v.to(device) for k, v in image.items()}
            target = target.to(device)
            feats.append(self._extract_features(image).cpu())
            labels.append(target.cpu())

        self._bank        = torch.cat(feats).to(device)       # (N, D)
        self._bank_labels = torch.cat(labels).to(device)      # (N,)
        
    @torch.no_grad()
    def classify(self, img: dict[str, Tensor]) -> Tensor:
        if self._bank is None:
            raise RuntimeError("Feature bank empty – call build_feature_bank() first")

        q = self._extract_features(img)
        return knn_predict(q, self._bank, self._bank_labels,
                           num_classes=self.num_classes,
                           knn_k=self.knn_k, knn_t=self.knn_t)      # (B, C) sorted
