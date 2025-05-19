import operator
import torch
import torch.nn.functional as F
from pangaea.engine.trainer import RegTrainer
from pangaea.engine.evaluator import RegEvaluator

class AGBDRegTrainer(RegTrainer):
    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Spatially average logits before MSE
        logits = torch.mean(logits, dim=(-2, -1))
        return self.criterion(logits, target)

    @torch.no_grad()
    def compute_logging_metrics(self, logits: torch.Tensor, target: torch.Tensor):
        logits = torch.mean(logits, dim=(-2, -1))
        mse = F.mse_loss(logits, target, reduction="mean").item()
        return {"MSE": mse}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric_comp = operator.lt  # For MSE minimization

class AGBDRegEvaluator(RegEvaluator):
    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        self.model = model
        self.model.eval()
        mse_sum = 0.0
        n_samples = 0
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)
            logits = torch.mean(logits, dim=(-2, -1))
            mse_sum += F.mse_loss(logits, labels, reduction="sum").item()
            n_samples += labels.size(0)
        final_mse = mse_sum / n_samples if n_samples > 0 else float('nan')
        metrics = {"MSE": final_mse}
        self.log_metrics(metrics)
        return metrics