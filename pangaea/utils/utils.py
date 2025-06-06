import os as os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import logging

_log = logging.getLogger(__name__)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to make flops calculator work
def prepare_input(input_res):
    image = {}
    x1 = torch.FloatTensor(*input_res)
    # input_res[-2] = 2
    input_res = list(input_res)
    input_res[-3] = 2
    x2 = torch.FloatTensor(*tuple(input_res))
    image["optical"] = x1
    image["sar"] = x2
    return dict(img=image)


def _find_ckpt(exp_dir: str | Path, suffix: str) -> Optional[str]:
    """Return the *first* file that ends with `suffix`; None if nothing found."""
    exp_dir = Path(exp_dir)
    for fname in exp_dir.iterdir():
        if fname.name.endswith(suffix):
            return str(fname)
    # Nothing found – warn once.
    _log.warning(
        "No checkpoint matching '*%s' found in %s. "
        "If this was a k-NN probe (no training), you can ignore this warning. Otherwise, check your experiment directory.",
        suffix, exp_dir,
    )
    return None


def get_best_model_ckpt_path(exp_dir: str | Path) -> Optional[str]:
    """Return '<exp_dir>/…_best.pth' or None when it does not exist."""
    return _find_ckpt(exp_dir, "_best.pth")


def get_final_model_ckpt_path(exp_dir: str | Path) -> Optional[str]:
    """Return '<exp_dir>/…_final.pth' or None when it does not exist."""
    return _find_ckpt(exp_dir, "_final.pth")