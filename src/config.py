import os
import time

from yacs.config import CfgNode as CN

_C = CN()

# Run config
_C.TAG = "default"
_C.OUTPUT_DIR = "artifacts"
_C.PRINT_EVERY = 10  # steps
_C.EVALUATE_EVERY = 5  # epochs
_C.SAVE_EVERY = 5  # epochs
_C.SAVE_TOP_K = 1  # checkpoints
_C.SEED = 2
_C.HOSTNAME = None

# Logging
_C.WANDB = CN()
_C.WANDB.PROJECT = "algtd"
_C.WANDB.ENABLED = False
_C.WANDB.OFFLINE = False

# Dataset
_C.DATASET = CN()
_C.DATASET.BASE_DIR = "data/gazefollow_extended"
_C.DATASET.NAME = "gazefollow"
_C.DATASET.SCENE_INPUT_SIZE = 224
_C.DATASET.FACE_INPUT_SIZE = 224
_C.DATASET.HEATMAP_OUTPUT_SIZE = 64
_C.DATASET.BATCH_SIZE = 16
_C.DATASET.NUM_WORKERS = max(
    1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() - 2)), 16)
)
_C.DATASET.LABELED_PATH = None  
_C.DATASET.UNLABELED_PATH = None
_C.DATASET.SEED = 89

# Model
_C.MODEL = CN()
_C.MODEL.NAME = "gtn"
_C.MODEL.PRETRAINED_WEIGHTS_PATH = None

# Training
_C.TRAINING = CN()
_C.TRAINING.DEVICE = "cuda"
_C.TRAINING.LR = 2.5e-4
_C.TRAINING.GRAD_CLIP = 5.0
_C.TRAINING.HM_LOSS_WEIGHT = 10_000
_C.TRAINING.INOUT_LOSS_WEIGHT = 0

# Active learning
_C.TRAINING.AL = CN()
_C.TRAINING.AL.EVAL_AFTER_ONE_EPOCH = False
_C.TRAINING.AL.STRATEGY = "al_gtd"
_C.TRAINING.AL.PERC_SAMPLES_LABELED = 0.03
_C.TRAINING.AL.PERC_SAMPLES_TO_LABEL_PER_CYCLE = 0.01
_C.TRAINING.AL.NUM_EPOCHS_PER_CYCLE = 15
_C.TRAINING.AL.NUM_CYCLES = 10
_C.TRAINING.AL.RELOAD_BEST_CHECKPOINT = True
_C.TRAINING.AL.RELOAD_BEST_CHECKPOINT_BY = "auc"
_C.TRAINING.AL.PERC_SAMPLES_TO_PSEUDO_LABEL_PER_CYCLE = 0.01
_C.TRAINING.AL.AL_GTD = CN()
_C.TRAINING.AL.AL_GTD.SALIENCY_WEIGHT = 1
_C.TRAINING.AL.AL_GTD.DISPERSION_WEIGHT = 1
_C.TRAINING.AL.AL_GTD.OBJECT_WEIGHT = 1
_C.TRAINING.AL.AL_GTD.DISTANCE_WEIGHT = 1

def get_config(
    config_path=None, config_args=None
):
    config = _C.clone()
    if config_path:
        config.merge_from_file(config_path)
    if config_args:
        config.merge_from_list(config_args)

    # Set hostname
    config.HOSTNAME = os.environ.get("HOSTNAME", None)

    # Set exp dir
    config.EXP_DIR = os.path.join(
        config.OUTPUT_DIR, config.TAG, f"seed_{config.SEED}"
    )
    os.makedirs(config.EXP_DIR, exist_ok=True)

    # Set run dir
    config.RUN_DIR = os.path.join(
        config.EXP_DIR,
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(config.RUN_DIR, exist_ok=False)

    # Print config tree
    config.freeze()
    print(config)
    print("")

    return config
