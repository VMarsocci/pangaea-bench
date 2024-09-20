import os as os
import pathlib
import pprint
import random
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from geofm_bench.engine.data_preprocessor import get_collate_fn
from geofm_bench.engine.trainer import Trainer
from geofm_bench.utils.logger import init_logger
from geofm_bench.utils.utils import fix_seed, get_generator, seed_worker


def get_exp_name(hydra_config: HydraConf) -> str:
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["foundation_model"]
    adaptor = choices["adaptor"]
    ds = choices["dataset"]
    return f"{timestamp}-{fm}-{adaptor}-{ds}"


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    exp_name = get_exp_name(HydraConfig.get())
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    if cfg.train:
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, config_log_dir / "config.yaml")
    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    # init wandb
    if cfg.use_wandb and rank == 0:
        import wandb

        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="geofm-bench",
            name=exp_name,
            config=wandb_cfg,
            resume="allow",
            id=cfg.get("wandb_run_id"),
        )
        # TODO: add wandb_run_id to the saved config
        # cfg["wandb_run_id"] = wandb.run.id

    # get datasets
    train_dataset: Dataset = instantiate(cfg.dataset, split="train")
    val_dataset: Dataset = instantiate(cfg.dataset, split="val")
    test_dataset: Dataset = instantiate(cfg.dataset, split="test")
    logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

    # Apply data processing to the datasets
    # for augmentation in cfg.augmentation.train:
    #     # TODO: add data augmentation cleaner so it can be implemented
    #     augmentation: RichDataset = instantiate(augmentation, cfg)
    #     print(augmentation)
    #     train_dataset = AUGMENTER_REGISTRY.get(step)(
    #         train_dataset, cfg, cfg.augmentation.train[step]
    #     )
    #
    # for step in cfg.augmentation.test:
    #     val_dataset = AUGMENTER_REGISTRY.get(step)(
    #         val_dataset, cfg, cfg.augmentation.test[step]
    #     )
    #     test_dataset = AUGMENTER_REGISTRY.get(step)(
    #         test_dataset, cfg, cfg.augmentation.test[step]
    #     )
    #
    # logger.info("Created processing pipelines:")
    # logger.info(f"   Training: {pprint.pformat([s for s in cfg.augmentation.train])}")
    # logger.info(f"   Evaluation: {pprint.pformat([s for s in cfg.augmentation.test])}")
    # #

    # prepare the foundation model
    # TODO: change download model signature with args
    # download_model(cfg.encoder)

    foundation_model: torch.nn.Module = instantiate(cfg.foundation_model)
    print(foundation_model)

    missing, incompatible_shape = foundation_model.load_encoder_weights()
    # TODO: refactor this part in load_encoder_weights(logger)
    # logger.info("Loaded encoder weight from {}.".format(cfg.encoder.encoder_weights))
    # if missing:
    #     logger.warning(
    #         "Missing parameters:\n"
    #         + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
    #     )
    # if incompatible_shape:
    #     logger.warning(
    #         "Incompatible parameters:\n"
    #         + "\n".join(
    #             "%s: expected %s but found %s" % (k, v[0], v[1])
    #             for k, v in sorted(incompatible_shape.items())
    #         )
    #     )
    #
    # prepare the adaptor (segmentation/regression)
    model: torch.nn.Module = instantiate(
        cfg.adaptor,
        encoder=foundation_model,
    )
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    logger.info(
        "Built {} for with {} encoder.".format(
            model.module.model_name, foundation_model.model_name
        )
    )

    modalities = list(foundation_model.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if cfg.train:
        if 0 < cfg.limited_label < 1:
            n_train_samples = len(train_dataset)
            indices = random.sample(
                range(n_train_samples), int(n_train_samples * cfg.limited_label)
            )
            train_dataset = Subset(train_dataset, indices)
            logger.info(
                f"Created a subset of the train dataset, with {cfg.limited_label * 100}% of the labels available"
            )
        else:
            logger.info("The entire train dataset will be used.")

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,  # cfg.dataset["batch"],
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # TODO: add val_evaluator in configs
        trainer: Trainer = instantiate(
            cfg.trainer,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            exp_dir=exp_dir,
            device=device,
        )


#
#         # training: put all components into engines
#         if task_name == "regression":
#             val_evaluator = RegEvaluator(cfg, val_loader, exp_dir, device)
#             trainer = RegTrainer(
#                 cfg,
#                 model,
#                 train_loader,
#                 criterion,
#                 optimizer,
#                 scheduler,
#                 val_evaluator,
#                 exp_dir,
#                 device,
#             )
#         else:
#             val_evaluator = SegEvaluator(cfg, val_loader, exp_dir, device)
#             trainer = SegTrainer(
#                 cfg,
#                 model,
#                 train_loader,
#                 criterion,
#                 optimizer,
#                 scheduler,
#                 val_evaluator,
#                 exp_dir,
#                 device,
#             )
#
#         # resume training if model_checkpoint is provided
#         if cfg.resume_from is not None:
#             trainer.load_model(cfg.resume_from)
#
#         trainer.train()
#
#     # Evaluation
#     else:
#         test_loader = DataLoader(
#             test_dataset,
#             sampler=DistributedSampler(test_dataset),
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             pin_memory=True,
#             persistent_workers=False,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )
#
#         logger.info("Built {} dataset for evaluation.".format(dataset_name))
#
#         if task_name == "regression":
#             # TODO: This doesn't work atm
#             test_evaluator = RegEvaluator(cfg, test_loader, exp_dir, device)
#         else:
#             test_evaluator = SegEvaluator(cfg, test_loader, exp_dir, device)
#
#         model_ckpt_path = os.path.join(
#             exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_best.pth"))
#         )
#         test_evaluator.evaluate(model, "best model", model_ckpt_path)
#
#     if cfg.use_wandb and cfg.rank == 0:
#         wandb.finish()
#
#
if __name__ == "__main__":
    main()