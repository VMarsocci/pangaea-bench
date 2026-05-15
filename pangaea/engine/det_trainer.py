import operator
import pathlib
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pangaea.engine.trainer import Trainer
from pangaea.utils.logger import RunningAverageMeter


class DetTrainer(Trainer):
    """Object detection trainer. Forward returns torchvision's loss dict; we sum it and log sub-losses."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module | None,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
        wandb_offline: bool = False,
        wandb_offline_sync_communication_dir: str | pathlib.Path = "",
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
            wandb_offline=wandb_offline,
            wandb_offline_sync_communication_dir=wandb_offline_sync_communication_dir,
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100)
            for name in [
                "loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg",
            ]
        }
        self.best_metric = float("-inf")
        self.best_metric_comp = operator.gt

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            image_dict = {
                modality: value.to(self.device)
                for modality, value in data["image"].items()
            }
            targets = [
                {
                    "boxes": data["boxes"][i].to(self.device),
                    "labels": data["labels"][i].to(self.device),
                }
                for i in range(len(data["boxes"]))
            ]

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                loss_dict = self.model(image_dict, targets)
                loss = sum(loss_dict.values())

            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats["loss"].update(loss.item())

            with torch.no_grad():
                for name, value in loss_dict.items():
                    if name in self.training_metrics:
                        self.training_metrics[name].update(value.item())

            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.item() for k, v in loss_dict.items()
                        },
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )
                if self.wandb_offline:
                    try:
                        self.trigger_sync()
                    except OSError as e:
                        self.logger.error(f"Error syncing Weights and Biases: {e}")

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()
