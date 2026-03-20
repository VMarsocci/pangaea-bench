#!/usr/bin/env python3
import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def build_command(args: argparse.Namespace, extra_overrides: list[str]) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    run_py = repo_root / "pangaea" / "run.py"

    cmd = [
        "torchrun",
        f"--nnodes={args.nnodes}",
        f"--nproc_per_node={args.nproc_per_node}",
    ]

    if args.master_port is not None:
        cmd.append(f"--master_port={args.master_port}")

    cmd.extend(
        [
            str(run_py),
            "--config-name=test",
            f"ckpt_dir={args.ckpt_dir}",
            f"cross_test_dataset={args.target_dataset}",
        ]
    )

    if args.dataset_root is not None:
        cmd.append(f"cross_test_root_path={args.dataset_root}")

    if args.preprocessing is not None:
        cmd.append(f"cross_test_preprocessing={args.preprocessing}")

    if args.test_batch_size is not None:
        cmd.append(f"test_batch_size={args.test_batch_size}")

    if args.test_num_workers is not None:
        cmd.append(f"test_num_workers={args.test_num_workers}")

    if args.use_final_ckpt:
        cmd.append("use_final_ckpt=true")

    if args.test_log_name is not None:
        cmd.append(f"test_log_name={args.test_log_name}")
    else:
        dataset_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.target_dataset).strip("_")
        if not dataset_slug:
            dataset_slug = "dataset"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        cmd.append(f"test_log_name=test_{dataset_slug}_{timestamp}.log")

    cmd.extend(extra_overrides)
    return cmd


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint directory on a different dataset "
            "(cross-testing wrapper for pangaea/run.py)."
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        help="Path to training experiment directory containing checkpoints and configs/config.yaml",
    )
    parser.add_argument(
        "--target-dataset",
        required=True,
        help="Dataset config name in configs/dataset (e.g., mados, hlsburnscars)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional override for target dataset root path",
    )
    parser.add_argument(
        "--preprocessing",
        default=None,
        help="Optional preprocessing config name in configs/preprocessing",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="torchrun nnodes")
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, help="torchrun nproc_per_node"
    )
    parser.add_argument(
        "--master-port", type=int, default=None, help="Optional torchrun master_port"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=None, help="Override test batch size"
    )
    parser.add_argument(
        "--test-num-workers", type=int, default=None, help="Override test dataloader workers"
    )
    parser.add_argument(
        "--use-final-ckpt",
        action="store_true",
        help="Use *_final.pth instead of *_best.pth",
    )
    parser.add_argument(
        "--test-log-name",
        default=None,
        help=(
            "Optional test log filename written inside --ckpt-dir "
            "(e.g., test_hlsburnscars.log). If omitted, a dataset-specific "
            "timestamped name is generated automatically."
        ),
    )

    args, extra_overrides = parser.parse_known_args()
    return args, extra_overrides


def main() -> int:
    args, extra_overrides = parse_args()

    command = build_command(args, extra_overrides)
    print("Running command:")
    print(" ".join(shlex.quote(c) for c in command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
