import argparse
import copy
import os

import rootutils
import torch
from dotenv import load_dotenv

import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.al_strategies import get_strategy  # noqa: E402
from src.config import get_config  # noqa: E402
from src.engine import evaluate, get_predictions_and_run_al, train  # noqa: E402
from src.models import get_model  # noqa: E402
from src.optimizer import get_optimizer  # noqa: E402
from src.utils.misc import (  # noqa: E402
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def main(config):
    # Set seed
    seed_everything(config.SEED)

    # Define device
    device = torch.device(config.TRAINING.DEVICE)
    print(f"Running on {device}")

    # Load model
    print("Loading model")
    model = get_model(config, device)

    # Get strategy
    print("Loading strategy")
    al_strategy = get_strategy(config)

    # Get optimizer
    optimizer = get_optimizer(config, model)
    optimizer.zero_grad()

    # Status variables
    cycle_idx = 0
    epoch_idx = 1

    # Load weights if needed
    if config.MODEL.PRETRAINED_WEIGHTS_PATH:
        print("Loading pretrained weights")
        load_checkpoint(
            model,
            torch.load(config.MODEL.PRETRAINED_WEIGHTS_PATH, map_location="cpu"),
        )

    # Set wandb
    if config.WANDB.ENABLED:
        wandb_logger = wandb.init(
            project=config.WANDB.PROJECT
        )
    else:
        wandb_logger = None

    # Keep track of current saved checkpoints
    saved_checkpoints = {}

    while al_strategy.is_valid_cycle(cycle_idx):
        print(f"Starting cycle {cycle_idx}")

        # Keep track of the best checkpoint of this cycle
        best_checkpoint = None
        best_checkpoint_auc = (
            10_000 if config.TRAINING.AL.RELOAD_BEST_CHECKPOINT_BY == "dist" else 0
        )

        print(f"Training for {al_strategy.num_epochs_per_cycle} epochs")

        # Train and evaluate
        for epoch_idx in range(epoch_idx, al_strategy.num_epochs_per_cycle + 1):
            train_loader, train_sampler = al_strategy.get_train_loader()

            # Training
            losses = train(
                config,
                model,
                device,
                train_loader,
                optimizer,
                cycle_idx,
                epoch_idx,
                al_strategy,
                wandb_logger=wandb_logger,
            )

            if config.WANDB.ENABLED:
                global_epoch = epoch_idx + (
                    cycle_idx * al_strategy.num_epochs_per_cycle
                )
                metrics = {
                    "al/epoch": epoch_idx,
                    "al/cycle": cycle_idx,
                    "al/training_samples": al_strategy.get_training_size(),
                    "al/labeled_samples": al_strategy.get_labeled_size(),
                    "al/unlabeled_samples": al_strategy.get_unlabeled_size(),
                    "al/pseudo_labeled_samples": al_strategy.get_pseudo_labeled_size(),
                    "epoch": global_epoch,
                }
                prefix = "train/epoch"
                for k, v in losses.items():
                    metrics[f"{prefix}/{k}"] = v

                wandb_logger.log(metrics)

            # Save
            if config.SAVE_EVERY > 0 and (
                epoch_idx % config.SAVE_EVERY == 0
                or epoch_idx == al_strategy.num_epochs_per_cycle
            ):
                save_checkpoint(
                    os.path.join(config.RUN_DIR, "last.ckpt"),
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "al_strategy": al_strategy,
                        "epoch_idx": epoch_idx,
                        "cycle_idx": cycle_idx,
                        "config": config,
                    },
                )

            # Evaluate
            if (
                (config.TRAINING.AL.EVAL_AFTER_ONE_EPOCH and epoch_idx == 1)
                or epoch_idx % config.EVALUATE_EVERY == 0
                or epoch_idx == al_strategy.num_epochs_per_cycle
            ):
                evals = evaluate(
                    config,
                    model,
                    device,
                    al_strategy.get_test_loader(),
                    cycle_idx,
                    epoch_idx,
                )

                if config.TRAINING.AL.RELOAD_BEST_CHECKPOINT_BY == "auc" and (
                    evals["auc"] > best_checkpoint_auc
                    and config.TRAINING.AL.RELOAD_BEST_CHECKPOINT
                ):
                    best_checkpoint = copy.deepcopy(model.state_dict())
                    best_checkpoint_auc = evals["auc"]

                sorted_checkpoints = sorted(
                    saved_checkpoints.items(), key=lambda kv: kv[1], reverse=False
                )

                if len(saved_checkpoints) >= config.SAVE_TOP_K and (
                    evals["auc"] > sorted_checkpoints[-1][1]
                ):
                    print(
                        f"Deleting checkpoint {sorted_checkpoints[-1][0]} with {config.TRAINING.AL.RELOAD_BEST_CHECKPOINT_BY} {sorted_checkpoints[-1][1]}"
                    )

                    delete_checkpoint(sorted_checkpoints[-1][0])
                    del saved_checkpoints[sorted_checkpoints[-1][0]]

                # Save the new checkpoint
                if len(saved_checkpoints) < config.SAVE_TOP_K:
                    store_metric = round(evals["auc"], 4)
                    checkpoint_path = os.path.join(
                        config.RUN_DIR,
                        f"cycle_{cycle_idx}_epoch_{epoch_idx}_auc_{store_metric}.pt",
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {"model": model.state_dict()},
                    )
                    saved_checkpoints[checkpoint_path] = store_metric

                if config.WANDB.ENABLED:
                    global_epoch = epoch_idx + (
                        cycle_idx * al_strategy.num_epochs_per_cycle
                    )
                    metrics = {
                        "al/epoch": epoch_idx,
                        "al/cycle": cycle_idx,
                        "al/training_samples": al_strategy.get_training_size(),
                        "al/labeled_samples": al_strategy.get_labeled_size(),
                        "al/unlabeled_samples": al_strategy.get_unlabeled_size(),
                        "al/pseudo_labeled_samples": al_strategy.get_pseudo_labeled_size(),
                        "epoch": global_epoch,
                    }
                    prefix = "test/epoch"
                    for k, v in evals.items():
                        metrics[f"{prefix}/{k}"] = v

                    wandb_logger.log(
                        metrics,
                    )

        # Reload best checkpoint if needed
        if best_checkpoint is not None and config.TRAINING.AL.RELOAD_BEST_CHECKPOINT:
            print(
                f"Reloading best checkpoint with {config.TRAINING.AL.RELOAD_BEST_CHECKPOINT_BY} {best_checkpoint_auc}"
            )
            load_checkpoint(model, {"model": best_checkpoint})

        # Save checkpoint before AL
        save_checkpoint(
            f"{config.RUN_DIR}/al_cycle_{cycle_idx}.ckpt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "al_strategy": al_strategy,
                "epoch_idx": epoch_idx,
                "cycle_idx": cycle_idx,
                "config": config,
            },
        )

        # Do AL
        get_predictions_and_run_al(
            config, model, device, al_strategy, cycle_idx, epoch_idx
        )
        al_strategy.update(cycle_idx)

        # Reload model
        print("Reloading model")
        model = get_model(config, device)

        # Load weights if needed
        if config.MODEL.PRETRAINED_WEIGHTS_PATH:
            print("Loading pretrained weights")
            load_checkpoint(
                model,
                torch.load(config.MODEL.PRETRAINED_WEIGHTS_PATH, map_location="cpu"),
            )

        optimizer = get_optimizer(config, model)
        cycle_idx += 1
        al_strategy.cycle_idx = cycle_idx
        epoch_idx = 1

if __name__ == "__main__":
    # Init everything
    torch.set_float32_matmul_precision("high")
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    # Read known and unknown arguments
    args, opts = parser.parse_known_args()

    # Ok let's go
    main(get_config(args.config, opts))
