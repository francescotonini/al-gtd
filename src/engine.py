import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from src.utils.gaze_ops import (
    get_angular_error,
    get_ap,
    get_heatmap_auc,
    get_heatmap_peak_coords,
    get_l2_dist,
    get_onehot_tgt_heatmap
)
from src.utils.metric_logger import MetricLogger


def train(
    config,
    model,
    device,
    loader,
    optimizer,
    cycle,
    epoch,
    al_strategy,
    wandb_logger=None,
):
    # Set model to train mode
    model.train()

    # Define metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f"TRAIN - [cycle {cycle} - epoch {epoch}]"

    for batch_idx, batch in metric_logger.log_every(loader, config.PRINT_EVERY, header):
        (
            tgt_rgb,
            tgt_depth,
            tgt_heads,
            _,  # tgt_heads_bbox
            tgt_heads_masks,
            tgt_gaze_heatmaps,
            tgt_eyes_coords,
            tgt_gaze_points,
            tgt_gaze_inside,
            _,  # img_size
            _,  # samples_key
            tgt_is_labelled,
            _,  # object_bboxes
            _,  # object_class
            _,  # object_confs
        ) = batch
        tgt_rgb = tgt_rgb.to(device, non_blocking=True)
        tgt_depth = tgt_depth.to(device, non_blocking=True)
        tgt_heads = tgt_heads.to(device, non_blocking=True)
        tgt_heads_masks = tgt_heads_masks.to(device, non_blocking=True)
        tgt_gaze_heatmaps = tgt_gaze_heatmaps.to(device, non_blocking=True)
        tgt_eyes_coords = tgt_eyes_coords.to(device, non_blocking=True).float()
        tgt_gaze_points = tgt_gaze_points.to(device, non_blocking=True).float()
        tgt_gaze_inside = tgt_gaze_inside.to(device, non_blocking=True).float()
        tgt_is_labelled = tgt_is_labelled.to(device, non_blocking=True).float()
        tgt_rgb = torch.cat([tgt_rgb, torch.flip(tgt_rgb, dims=[3])], dim=0)
        tgt_depth = torch.cat([tgt_depth, torch.flip(tgt_depth, dims=[3])], dim=0)
        tgt_heads = torch.cat([tgt_heads, torch.flip(tgt_heads, dims=[3])], dim=0)
        tgt_heads_masks = torch.cat(
            [tgt_heads_masks, torch.flip(tgt_heads_masks, dims=[2])], dim=0
        )
        tgt_gaze_heatmaps_flipped = torch.flip(tgt_gaze_heatmaps, dims=[2])
        tgt_gaze_heatmaps = torch.cat([tgt_gaze_heatmaps, tgt_gaze_heatmaps_flipped], dim=0)
        tgt_gaze_points_flipped = tgt_gaze_points.clone()
        tgt_gaze_points_flipped[:, 0] = 1 - tgt_gaze_points_flipped[:, 0]
        tgt_gaze_points = torch.cat([tgt_gaze_points, tgt_gaze_points_flipped], dim=0)
        tgt_gaze_inside = torch.cat([tgt_gaze_inside, tgt_gaze_inside], dim=0)
        tgt_is_labelled = torch.cat([tgt_is_labelled, tgt_is_labelled], dim=0)
        tgt_gaze_presence = torch.mul(tgt_gaze_inside, tgt_is_labelled)

        pred_gaze_heatmaps, pred_gaze_inside, pred_saliency = model(
            tgt_rgb, tgt_depth, tgt_heads, tgt_heads_masks
        )
        pred_saliency = pred_saliency.squeeze(1)
        pred_gaze_inside = pred_gaze_inside.squeeze(1)

        # Keep track of all losses
        losses = {}

        # Heatmap loss
        pred_gaze_heatmaps = pred_gaze_heatmaps.squeeze(1)
        hm_loss = (
            torch.nn.functional.mse_loss(
                pred_gaze_heatmaps, tgt_gaze_heatmaps, reduction="none"
            )
            * config.TRAINING.HM_LOSS_WEIGHT
        )
        hm_loss = torch.mean(hm_loss, dim=1)
        hm_loss = torch.mean(hm_loss, dim=1)
        losses["hm_loss"] = hm_loss.mean()

        # Inout loss
        inout_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pred_gaze_inside, tgt_gaze_presence
            )
            * config.TRAINING.INOUT_LOSS_WEIGHT
        )
        losses["inout_loss"] = inout_loss

        # SSL loss
        dim_to_flip = 2
        pred_gaze_heatmaps_orig = pred_gaze_heatmaps[: pred_gaze_heatmaps.shape[0] // 2]
        pred_gaze_heatmaps_flipped = torch.flip(
            pred_gaze_heatmaps[pred_gaze_heatmaps.shape[0] // 2 :], dims=[dim_to_flip]
        )

        hm_ssl_loss = torch.nn.functional.mse_loss(
            pred_gaze_heatmaps_orig, pred_gaze_heatmaps_flipped
        ) * (config.TRAINING.HM_LOSS_WEIGHT // 100)
        losses["hm_ssl_loss"] = hm_ssl_loss

        loss = sum(losses.values())

        # Stop if loss is nan
        if not torch.isfinite(loss):
            print(losses)
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)

        # Backprop
        loss.backward()

        # Gradient clipping
        if config.TRAINING.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING.GRAD_CLIP)

        optimizer.step()
        optimizer.zero_grad()

        # Update metric logger
        metric_logger.update(**losses)

        # Log wandb
        if config.WANDB.ENABLED:
            global_epoch = epoch + (cycle * al_strategy.num_epochs_per_cycle)
            metrics = {
                "al/cycle": cycle,
                "al/epoch": epoch,
                "epoch": global_epoch,
            }
            prefix = "train/step"
            for key, value in losses.items():
                metrics[f"{prefix}/{key}"] = value

            wandb_logger.log(metrics)

    # Gather and return the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(config, model, device, loader, cycle, epoch):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    tgt_gaze_inside_all = []
    pred_gaze_inside_all = []
    header = f"TEST - [cycle {cycle} - epoch {epoch}]"

    for batch_idx, batch in metric_logger.log_every(loader, config.PRINT_EVERY, header):
        (
            tgt_rgb,
            tgt_depth,
            tgt_heads,
            tgt_heads_bbox,
            tgt_heads_masks,
            tgt_gaze_heatmaps,
            tgt_eyes_coords,
            tgt_gaze_points,
            tgt_gaze_inside,
            img_size,
            samples_key,
            _,  # is_labelled
            _,  # object_bboxes
            _,  # object_class
            _,  # object_confs
        ) = batch

        tgt_rgb = tgt_rgb.to(device, non_blocking=True)
        tgt_depth = tgt_depth.to(device, non_blocking=True)
        tgt_heads = tgt_heads.to(device, non_blocking=True)
        tgt_heads_masks = tgt_heads_masks.to(device, non_blocking=True)
        tgt_eyes_coords = tgt_eyes_coords.to(device, non_blocking=True)
        tgt_gaze_points = tgt_gaze_points.to(device, non_blocking=True)
        tgt_gaze_inside = tgt_gaze_inside.to(device, non_blocking=True)
        img_size = img_size.to(device, non_blocking=True)

        # Get predictions
        with torch.no_grad():
            pred_gaze_heatmaps, pred_gaze_inside, _ = model(
                tgt_rgb, tgt_depth, tgt_heads, tgt_heads_masks
            )

        # Move to cpu
        tgt_rgb = tgt_rgb.cpu()
        tgt_depth = tgt_depth.cpu()
        tgt_heads = tgt_heads.cpu()
        tgt_heads_masks = tgt_heads_masks.cpu()
        tgt_eyes_coords = tgt_eyes_coords.cpu()
        tgt_gaze_points = tgt_gaze_points.cpu()
        tgt_gaze_inside = tgt_gaze_inside.cpu()
        img_size = img_size.cpu()
        pred_gaze_heatmaps = pred_gaze_heatmaps.cpu()
        pred_gaze_inside = pred_gaze_inside.cpu()

        # Store gaze inside ground truth and predictions for later evaluation
        tgt_gaze_inside_all.extend(tgt_gaze_inside.tolist())
        pred_gaze_inside_all.extend(pred_gaze_inside.squeeze(1).tolist())

        metrics = []
        for b_i in range(len(tgt_gaze_points)):
            metrics.append(
                evaluate_one_item(
                    config,
                    samples_key[0][b_i],
                    tgt_rgb[b_i],
                    pred_gaze_heatmaps[b_i],
                    tgt_gaze_heatmaps[b_i],
                    tgt_eyes_coords[b_i],
                    tgt_gaze_points[b_i],
                    img_size[b_i],
                    tgt_heads_bbox[b_i],
                )
            )

        for metric in metrics:
            if metric is None:
                continue

            (
                auc_score,
                min_dist,
                avg_dist,
                min_ang_err,
                avg_ang_err,
            ) = metric

            metric_logger.update(
                auc=auc_score,
                min_dist=min_dist,
                avg_dist=avg_dist,
                min_ang_err=min_ang_err,
                avg_ang_err=avg_ang_err,
            )

    # Add gaze inside AP
    metric_logger.update(
        gaze_inside_ap=get_ap(
            torch.tensor(pred_gaze_inside_all),
            torch.tensor(tgt_gaze_inside_all),
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_one_item(
    config,
    name,
    tgt_rgb,
    pred_gaze_heatmap,
    real_tgt_gaze_heatmap,
    tgt_eye_points,
    tgt_gaze_points,
    img_size,
    tgt_head_bbox,
):
    # Remove padding and recover valid ground truth points
    tgt_gaze_points = tgt_gaze_points[tgt_gaze_points >= 0].view(-1, 2)
    tgt_eye_points = tgt_eye_points[tgt_eye_points >= 0].view(-1, 2)

    # Skip items that do not have valid gaze coords
    if len(tgt_gaze_points) == 0:
        print("Skipping item with no gaze points")

        return

    # AUC: area under the curve of the ROC curve across predicted gaze heatmap
    # and ground truth heatmap
    pred_gaze_heatmap_rescaled = TF.resize(
        pred_gaze_heatmap,
        (img_size[0], img_size[1]),
        antialias=True,
    ).squeeze()
    tgt_gaze_heatmap = get_onehot_tgt_heatmap(tgt_gaze_points, img_size)
    auc = get_heatmap_auc(pred_gaze_heatmap_rescaled, tgt_gaze_heatmap)

    # Get peak coords of the predicted gaze heatmap
    pred_gaze_point = get_heatmap_peak_coords(pred_gaze_heatmap.squeeze())

    # Collect distances and angular errors between the predicted point and human gaze points
    all_gaze_distances = get_l2_dist(pred_gaze_point, tgt_gaze_points)
    all_gaze_angular_errors = get_angular_error(
        pred_gaze_point - tgt_eye_points,
        tgt_gaze_points - tgt_eye_points,
    )

    min_gaze_distance = torch.min(all_gaze_distances)
    min_gaze_angular_error = torch.min(all_gaze_angular_errors)
    avg_tgt_gaze_point = torch.mean(tgt_gaze_points, dim=0, keepdim=True)
    avg_gaze_distance = get_l2_dist(avg_tgt_gaze_point, pred_gaze_point)
    avg_gaze_angular_error = get_angular_error(
        avg_tgt_gaze_point - torch.mean(tgt_eye_points, dim=0, keepdim=True),
        pred_gaze_point - torch.mean(tgt_eye_points, dim=0, keepdim=True),
    )

    return (
        auc,
        min_gaze_distance,
        avg_gaze_distance,
        min_gaze_angular_error,
        avg_gaze_angular_error,
    )


def get_predictions_and_run_al(
    config,
    model,
    device,
    al_strategy,
    cycle,
    epoch,
):
    # Set eval mode
    model.eval()

    # Init logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f"AL - [cycle {cycle}]"

    # Iterate
    for batch_idx, batch in metric_logger.log_every(
        al_strategy.get_unlabeled_loader(), config.PRINT_EVERY, header
    ):
        (
            tgt_rgb,
            tgt_depth,
            tgt_heads,
            _,  # tgt_heads_bbox
            tgt_heads_masks,
            tgt_gaze_heatmaps,
            tgt_eyes_coords,
            _,  # tgt_gaze_points,
            _,  # tgt_gaze_inside,
            _,  # img_size,
            samples_key,
            _,  # is_labelled
            object_bboxes,
            object_class,
            object_confs,
        ) = batch

        samples_key = [s for s in zip(samples_key[0], samples_key[1].tolist())]
        tgt_rgb = tgt_rgb.to(device, non_blocking=True)
        tgt_depth = tgt_depth.to(device, non_blocking=True)
        tgt_heads = tgt_heads.to(device, non_blocking=True)
        tgt_heads_masks = tgt_heads_masks.to(device, non_blocking=True)
        tgt_eyes_coords = tgt_eyes_coords.to(device, non_blocking=True)
        tgt_gaze_heatmaps = tgt_gaze_heatmaps.to(device, non_blocking=True)
        samples_key = samples_key + samples_key
        tgt_rgb = torch.cat([tgt_rgb, torch.flip(tgt_rgb, dims=[3])], dim=0)
        tgt_depth = torch.cat([tgt_depth, torch.flip(tgt_depth, dims=[3])], dim=0)
        tgt_heads = torch.cat([tgt_heads, torch.flip(tgt_heads, dims=[3])], dim=0)
        tgt_heads_masks = torch.cat(
            [tgt_heads_masks, torch.flip(tgt_heads_masks, dims=[2])], dim=0
        )
        tgt_gaze_heatmaps = torch.cat(
            [tgt_gaze_heatmaps, torch.flip(tgt_gaze_heatmaps, dims=[2])], dim=0
        )

        with torch.no_grad():
            pred_gaze_heatmaps, pred_gaze_inside, pred_saliency = model(
                tgt_rgb,
                tgt_depth,
                tgt_heads,
                tgt_heads_masks
            )

        pred_gaze_heatmaps = pred_gaze_heatmaps.squeeze(1)
        pred_gaze_inside = pred_gaze_inside.squeeze(1)
        pred_saliency = pred_saliency.squeeze(1)

        preds = {
            "samples_key": samples_key,
            "pred_gaze_heatmap": pred_gaze_heatmaps.float().cpu(),
            "object_bboxes": object_bboxes,
            "object_class": object_class,
            "object_confs": object_confs,
            "tgt_gaze_heatmap": tgt_gaze_heatmaps.float().cpu(),
            "pred_saliency": pred_saliency.squeeze(1).float().cpu(),
        }
        al_strategy.score_predictions(preds)
