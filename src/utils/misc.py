import os
import random

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def load_checkpoint(model, checkpoint, drop_prefix=None):
    model_dict = model.state_dict()
    model_weights = (
        checkpoint["model"]
        if "model" in checkpoint
        else (
            checkpoint["state_dict"]
            if "state_dict" in checkpoint
            else checkpoint["model_state_dict"]
        )
    )

    new_state_dict = {}
    for k, v in model_weights.items():
        if drop_prefix is not None and k.startswith(drop_prefix):
            k = k[len(drop_prefix) :]

        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
        elif k in model_dict and model_dict[k].shape != v.shape:
            print(
                f"Skipping {k} from pretrained weights: shape mismatch ({v.shape} vs {model_dict[k].shape})"
            )
        else:
            print(f"Skipping {k} from pretrained weights: not found in model")

    print(f"Total weights from file: {len(model_weights)}")
    print(f"Total weights loaded: {len(new_state_dict)}")

    model_dict.update(new_state_dict)
    print(model.load_state_dict(model_dict, strict=False))


def save_checkpoint(save_path, checkpoint):
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def delete_checkpoint(save_path):
    os.remove(save_path)
    print(f"Checkpoint deleted at {save_path}")

