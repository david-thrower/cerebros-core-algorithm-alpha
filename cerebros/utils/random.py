import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import jax
        jax.random.PRNGKey(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
        if deterministic:
            # Best-effort determinism; may reduce performance
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except Exception:
        pass
