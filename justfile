set shell := ["bash", "-uc"]

default:
    @just --list

setup:
    uv sync --all-groups
    uv run pre-commit install --install-hooks

lint:
    uv run pre-commit run --all-files

test *ARGS:
    uv run pytest {{ARGS}}

build:
    uv build

run *ARGS:
    uv run torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
        --config-name=train \
        dataset=mados \
        encoder=prithvi \
        decoder=seg_upernet_mt_ltae \
        preprocessing=seg_default \
        criterion=cross_entropy \
        task=segmentation \
        {{ARGS}}
