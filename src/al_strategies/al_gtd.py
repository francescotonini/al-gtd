import math

import torch
from torch.utils.data import ConcatDataset

from src.utils import gaze_ops

from .base import BaseStrategy


class ALGTDStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

        self.num_samples_to_pseudo_label_per_cycle = int(
            (self.get_labeled_size() + self.get_pseudo_labeled_size() + self.get_unlabeled_size())
            * (self.config.TRAINING.AL.PERC_SAMPLES_TO_PSEUDO_LABEL_PER_CYCLE)
        )
        self.pseudo_labels_scores_tmp = {}
        self.pseudo_labels_tmp = {}

        self.init_config()

    def init_config(self):
        saliency_array = [self.config.TRAINING.AL.AL_GTD.SALIENCY_WEIGHT for _ in range(self.config.TRAINING.AL.NUM_CYCLES)]
        dispersion_array = [
            self.config.TRAINING.AL.AL_GTD.DISPERSION_WEIGHT for _ in range(self.config.TRAINING.AL.NUM_CYCLES)
        ]
        object_array = [self.config.TRAINING.AL.AL_GTD.OBJECT_WEIGHT for _ in range(self.config.TRAINING.AL.NUM_CYCLES)]
        distance_array = [self.config.TRAINING.AL.AL_GTD.DISTANCE_WEIGHT for _ in range(self.config.TRAINING.AL.NUM_CYCLES)]
        assert len(saliency_array) == self.config.TRAINING.AL.NUM_CYCLES
        assert len(dispersion_array) == self.config.TRAINING.AL.NUM_CYCLES
        assert len(object_array) == self.config.TRAINING.AL.NUM_CYCLES
        assert len(distance_array) == self.config.TRAINING.AL.NUM_CYCLES

        self.saliency_weight = saliency_array
        self.dispersion_weight = dispersion_array
        self.object_weight = object_array
        self.distance_weight = distance_array

    def get_saliency_score(self, heatmap, attn):
        # Rescale heatmap into 7x7
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(0),
            size=(7, 7),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Calculate peak saliency and heatmap point
        peak_saliency_coords = gaze_ops.get_heatmap_peak_coords_batched(attn) * 7
        peak_heatmap_coords = gaze_ops.get_heatmap_peak_coords_batched(heatmap) * 7

        # Calculate distance between peak saliency and peak heatmap
        dists = gaze_ops.get_l2_dist(
            peak_saliency_coords,
            peak_heatmap_coords,
        ) / math.sqrt(98)

        return dists

    def get_object_score(self, heatmaps, objects_bboxes, objects_class, objects_confs):
        scores = []
        for i in range(len(heatmaps)):
            object_idxs = torch.where(objects_confs[i] > 0.1)[0]
            object_bboxes = objects_bboxes[i][object_idxs]
            object_confs = objects_confs[i][object_idxs]
            heatmap = heatmaps[i].unsqueeze(0)

            score = 0
            if object_bboxes.numel() != 0:
                coords = gaze_ops.get_heatmap_peak_coords_batched(heatmap)
                object_inside_boxes = (
                    gaze_ops.points_inside_boxes(coords, object_bboxes) * object_confs
                )

                if object_inside_boxes.sum() == 0:
                    score = 0
                else:
                    og_objects_inside_boxes = object_inside_boxes
                    max_og_object = og_objects_inside_boxes.max(dim=1).values
                    score = max_og_object.mean()

            scores.append(score)

        return torch.tensor(scores)

    def get_dispersion_score(self, heatmaps):
        dispersions = []
        for heatmap in heatmaps:
            dists, _ = gaze_ops.get_uncertainty(
                heatmap,
                points_to_sample=5,
                reduction="none",
            )

            dispersion = dists.max()
            dispersions.append(dispersion)

        return torch.stack(dispersions)

    def score_predictions(self, predictions):
        samples_key = predictions["samples_key"]
        samples_key = samples_key[: len(samples_key) // 2]
        pred_gaze_heatmaps = predictions["pred_gaze_heatmap"]
        pred_saliency = predictions["pred_saliency"]
        object_bboxes = predictions["object_bboxes"]
        object_labels = predictions["object_class"]
        object_confs = predictions["object_confs"]
        dim_to_flip = 2
        pred_gaze_heatmaps_orig = pred_gaze_heatmaps[: len(pred_gaze_heatmaps) // 2]
        pred_saliency_orig = pred_saliency[: len(pred_gaze_heatmaps) // 2]
        pred_gaze_heatmaps_flipped = pred_gaze_heatmaps[len(pred_gaze_heatmaps) // 2 :]
        pred_gaze_heatmaps_flipped = torch.flip(
            pred_gaze_heatmaps[len(pred_gaze_heatmaps) // 2 :], dims=[dim_to_flip]
        )
        pred_saliency_flipped = pred_saliency[len(pred_gaze_heatmaps) // 2 :]
        pred_saliency_flipped = torch.flip(
            pred_saliency[len(pred_gaze_heatmaps) // 2 :], dims=[dim_to_flip]
        )

        # Dispersion score
        dispersion_scores_orig = self.get_dispersion_score(pred_gaze_heatmaps_orig)
        dispersion_scores_flipped = self.get_dispersion_score(pred_gaze_heatmaps_flipped)
        dispersion_scores = torch.max(dispersion_scores_orig, dispersion_scores_flipped)

        # Saliency score
        saliency_scores_orig = self.get_saliency_score(pred_gaze_heatmaps_orig, pred_saliency_orig)
        saliency_scores_flipped = self.get_saliency_score(
            pred_gaze_heatmaps_flipped, pred_saliency_flipped
        )
        saliency_scores = torch.max(saliency_scores_orig, saliency_scores_flipped)

        # Object score
        objects_scores_orig = self.get_object_score(
            pred_gaze_heatmaps_orig, object_bboxes, object_labels, object_confs
        )
        objects_scores_flipped = self.get_object_score(
            pred_gaze_heatmaps_flipped,
            object_bboxes,
            object_labels,
            object_confs,
        )
        objects_scores = torch.max(objects_scores_orig, objects_scores_flipped)

        pred_gaze_heatmaps = torch.cat(
            (pred_gaze_heatmaps_orig.unsqueeze(1), pred_gaze_heatmaps_flipped.unsqueeze(1)), dim=1
        )

        for _, (
            key,
            object_score,
            dispersion_score,
            saliency_score,
            pred_gaze_heatmap,
        ) in enumerate(
            zip(
                samples_key,
                objects_scores,
                dispersion_scores,
                saliency_scores,
                pred_gaze_heatmaps,
            )
        ):
            pred_peak_coords = gaze_ops.get_heatmap_peak_coords_batched(pred_gaze_heatmap)
            pred_peak = pred_gaze_heatmap.max(dim=2).values.max(dim=1).values
            coords_dist = gaze_ops.get_l2_dist(pred_peak_coords[0:1], pred_peak_coords[1:2])

            self.scores[key] = (
                self.saliency_weight[self.cycle_idx] * saliency_score
                + self.dispersion_weight[self.cycle_idx] * dispersion_score
                + self.object_weight[self.cycle_idx] * object_score
                + self.distance_weight[self.cycle_idx] * coords_dist
            ).item()

            self.pseudo_labels_scores_tmp[key] = pred_peak.max() * (1 - dispersion_score)
            self.pseudo_labels_tmp[key] = {
                "gaze_coords": pred_peak_coords.mean(dim=0),
            }

    def update(self, cycle_idx):
        self.scores = dict(sorted(self.scores.items(), key=lambda item: item[1], reverse=True))
        self.pseudo_labels_scores_tmp = dict(
            sorted(
                self.pseudo_labels_scores_tmp.items(),
                key=lambda item: item[1],
                reverse=False,
            )
        )

        num_samples_to_pseudo_label = min(
            self.num_samples_to_pseudo_label_per_cycle,
            len(self.scores),
        )
        num_samples_to_label = min(
            self.num_samples_to_label_per_cycle,
            len(self.scores),
        )

        sampled_keys = list(self.scores.keys())
        pseudo_samples = list(self.pseudo_labels_scores_tmp.keys())
        for key in pseudo_samples[:num_samples_to_pseudo_label]:
            self.pseudo_labeled_dataset.add_pseudo_annotations(
                key,
                {
                    "gaze_coords": self.pseudo_labels_tmp[key]["gaze_coords"],
                },
            )

        labeled_keys = self.labeled_dataset.get_keys()
        unlabeled_keys = self.unlabeled_dataset.get_keys()
        pseudo_labeled_keys = self.pseudo_labeled_dataset.get_keys()

        sampled_keys = [key for key in sampled_keys if key not in pseudo_labeled_keys]
        sampled_keys = sampled_keys[:num_samples_to_label]
        for key in sampled_keys:
            labeled_keys.append(key)
            unlabeled_keys.remove(key)
        for key in pseudo_labeled_keys:
            if key in unlabeled_keys:
                unlabeled_keys.remove(key)

        self.save_labeled_and_unlabeled(cycle_idx + 1)
        self.save_pseudo_labeled(cycle_idx + 1)
        self.save_scores(cycle_idx)
        self.reset_scores()

        self.pseudo_labels_tmp = {}
        self.pseudo_labels_scores_tmp = {}

    def get_train_dataset(self):
        return ConcatDataset([self.labeled_dataset, self.pseudo_labeled_dataset])
