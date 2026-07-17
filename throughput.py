"""Measure inference throughput (samples/sec) of a model on a given dataset.

Usage:
    python throughput.py \
        task=regression dataset=agbdlite encoder=gfmswin \
        decoder=reg_upernet preprocessing=reg_default criterion=mse \
        batch_size=64 \
        --warmup 20 --iterations 100

All Hydra overrides (task, dataset, encoder, etc.) work as in run.py.
Extra CLI flags (--warmup, --iterations) control the benchmark.
"""

import argparse
import sys
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from pangaea.datasets.base import GeoFMDataset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.utils.collate_fn import get_collate_fn


def parse_extra_args():
    """Parse --warmup and --iterations before Hydra consumes the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warm-up forward passes (discarded).")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of timed forward passes.")
    args, remaining = parser.parse_known_args()
    # Put remaining args back so Hydra can parse them
    sys.argv = [sys.argv[0]] + remaining
    return args


extra_args = parse_extra_args()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    warmup = extra_args.warmup
    iterations = extra_args.iterations
    batch_size = cfg.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("WARNING: No GPU detected. Throughput numbers on CPU are not meaningful.")

    # ── Build model ──────────────────────────────────────────────────────
    encoder: Encoder = instantiate(cfg.encoder)

    #if encoder.model_name != "Prithvi":
    #   encoder.load_encoder_weights(None)  # None logger → silent

    decoder: Decoder = instantiate(cfg.decoder, encoder=encoder)
    decoder.to(device)
    decoder.eval()

    n_params = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Model        : {decoder.model_name}  (encoder: {encoder.model_name})")
    print(f"Parameters   : {n_params:,} total, {n_trainable:,} trainable")

    # ── Build dataset & loader ───────────────────────────────────────────
    preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    raw_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    dataset = GeoFMDataset(raw_dataset, preprocessor)

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )

    print(f"Dataset      : {cfg.dataset.dataset_name}  (split=test, {len(dataset)} samples)")
    print(f"Batch size   : {batch_size}")
    print(f"Warm-up      : {warmup} forward passes")
    print(f"Iterations   : {iterations} forward passes")
    print()

    # ── Helper: get one batch (cycles the loader) ────────────────────────
    loader_iter = iter(loader)

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    # ── Warm-up ──────────────────────────────────────────────────────────
    print("Warming up …")
    with torch.no_grad():
        for _ in range(warmup):
            data = next_batch()
            image = {k: v.to(device, non_blocking=True) for k, v in data["image"].items()}
            target = data["target"].to(device, non_blocking=True)
            _ = decoder(image, output_shape=target.shape[-2:])

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Timed run ────────────────────────────────────────────────────────
    print("Benchmarking …")
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            data = next_batch()
            image = {k: v.to(device, non_blocking=True) for k, v in data["image"].items()}
            target = data["target"].to(device, non_blocking=True)
            _ = decoder(image, output_shape=target.shape[-2:])

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start

    # ── Report ───────────────────────────────────────────────────────────
    total_samples = iterations * batch_size
    throughput = total_samples / elapsed
    ms_per_sample = (elapsed / total_samples) * 1000
    ms_per_batch = (elapsed / iterations) * 1000

    print()
    print("═" * 50)
    print(f"  Total time     : {elapsed:.2f} s")
    print(f"  Batches        : {iterations}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Throughput     : {throughput:.1f} samples/s")
    print(f"  Latency/sample : {ms_per_sample:.2f} ms")
    print(f"  Latency/batch  : {ms_per_batch:.2f} ms")
    print("═" * 50)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"  Peak GPU mem   : {peak_mem:.2f} GB")
        print("═" * 50)


if __name__ == "__main__":
    main()
