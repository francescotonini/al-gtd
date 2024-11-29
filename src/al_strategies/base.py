import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.GazeFollow import GazeFollow
from src.datasets.PseudoGazeFollow import PseudoGazeFollow
from src.datasets.PseudoVideoAttentionTarget import PseudoVideoAttentionTarget
from src.datasets.transforms.ToColorMap import ToColorMap
from src.datasets.VideoAttentionTarget import VideoAttentionTarget


class BaseStrategy:
    def __init__(self, config):
        self.config = config

        # Prepare starting datasets
        self.labeled_dataset = self.get_dataset(
            config, "labeled", config.TRAINING.AL.PERC_SAMPLES_LABELED
        )
        self.pseudo_labeled_dataset = self.get_dataset(config, "pseudo_labeled", 1)
        self.unlabeled_dataset = self.get_dataset(
            config, "unlabeled", 1 - config.TRAINING.AL.PERC_SAMPLES_LABELED
        )
        self.test_dataset = self.get_dataset(config, "test", 1)

        # Check that labeled and unlabeled datasets are disjoint
        assert (
            set(self.labeled_dataset.get_keys()).intersection(
                set(self.unlabeled_dataset.get_keys()).intersection(
                    set(self.pseudo_labeled_dataset.get_keys())
                )
            )
            == set()
        )

        self.num_al_cycles = self.config.TRAINING.AL.NUM_CYCLES
        self.num_epochs_per_cycle = self.config.TRAINING.AL.NUM_EPOCHS_PER_CYCLE
        self.num_samples_to_label_per_cycle = int(
            (self.get_labeled_size() + self.get_pseudo_labeled_size() + self.get_unlabeled_size())
            * (self.config.TRAINING.AL.PERC_SAMPLES_TO_LABEL_PER_CYCLE)
        )
        if self.num_samples_to_label_per_cycle == 0:
            print(
                "WARNING: num_samples_to_label_per_cycle is 0, no active learning will be performed"
            )

        # Keep track of the scores assigned to each sample
        self.scores = {}

        # Save the initial state of the datasets
        self.cycle_idx = 0
        self.save_labeled_and_unlabeled(self.cycle_idx)

    def init_config(self):
        pass

    def get_transforms(self, config, scene_input_size, face_input_size):
        rgb_transform = transforms.Compose(
            [
                transforms.Resize((scene_input_size, scene_input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        depth_transform = transforms.Compose(
            [
                ToColorMap(plt.get_cmap("magma")),
                transforms.Resize((scene_input_size, scene_input_size), antialias=True),
                transforms.ToTensor(),
            ]
        )
        face_transform = transforms.Compose(
            [
                transforms.Resize((face_input_size, face_input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return rgb_transform, depth_transform, face_transform

    def get_dataset(self, config, subset="labeled", subset_size=0.1):
        rgb_transform, depth_transform, face_transform = self.get_transforms(
            config, self.config.DATASET.SCENE_INPUT_SIZE, self.config.DATASET.FACE_INPUT_SIZE
        )

        if self.config.DATASET.NAME == "gazefollow":
            dataset_fn = GazeFollow
            if subset == "pseudo_labeled":
                dataset_fn = PseudoGazeFollow

            dataset = dataset_fn(
                self.config.DATASET.BASE_DIR,
                seed=self.config.DATASET.SEED,
                input_size=self.config.DATASET.SCENE_INPUT_SIZE,
                output_size=self.config.DATASET.HEATMAP_OUTPUT_SIZE,
                subset=subset,
                subset_size=subset_size,
                rgb_transform=rgb_transform,
                face_transform=face_transform,
                depth_transform=depth_transform,
                override_keys_path=(
                    self.config.DATASET.LABELED_PATH
                    if subset == "labeled"
                    else self.config.DATASET.UNLABELED_PATH if subset == "unlabeled" else None
                ),
            )
        elif self.config.DATASET.NAME == "videoattentiontarget":
            dataset_fn = VideoAttentionTarget
            if subset == "pseudo_labeled":
                dataset_fn = PseudoVideoAttentionTarget

            dataset = dataset_fn(
                self.config.DATASET.BASE_DIR,
                seed=self.config.DATASET.SEED,
                input_size=self.config.DATASET.SCENE_INPUT_SIZE,
                output_size=self.config.DATASET.HEATMAP_OUTPUT_SIZE,
                subset=subset,
                subset_size=subset_size,
                rgb_transform=rgb_transform,
                face_transform=face_transform,
                depth_transform=depth_transform,
                override_keys_path=(
                    self.config.DATASET.LABELED_PATH
                    if subset == "labeled"
                    else self.config.DATASET.UNLABELED_PATH if subset == "unlabeled" else None
                ),
            )
        else:
            raise ValueError(f"Invalid dataset: {self.config.DATASET.NAME}")

        return dataset

    def get_train_dataset(self):
        return self.labeled_dataset

    def get_train_loader(self):
        sampler_train = torch.utils.data.RandomSampler(self.get_train_dataset())

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, batch_size=self.config.DATASET.BATCH_SIZE, drop_last=True
        )

        return (
            DataLoader(
                dataset=self.get_train_dataset(),
                batch_sampler=batch_sampler_train,
                num_workers=self.config.DATASET.NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=4 if self.config.DATASET.NUM_WORKERS > 0 else None,
            ),
            sampler_train,
        )

    def get_pseudo_labeled_loader(self):
        sampler_train = torch.utils.data.RandomSampler(self.pseudo_labeled_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, batch_size=self.config.DATASET.BATCH_SIZE, drop_last=True, shuffle=False
        )

        return (
            DataLoader(
                dataset=self.pseudo_labeled_dataset,
                batch_sampler=batch_sampler_train,
                num_workers=self.config.DATASET.NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=4 if self.config.DATASET.NUM_WORKERS > 0 else None,
            ),
            sampler_train,
        )

    def get_unlabeled_loader(self):
        sampler_al = torch.utils.data.RandomSampler(self.unlabeled_dataset)

        return DataLoader(
            dataset=self.unlabeled_dataset,
            batch_size=self.config.DATASET.BATCH_SIZE,
            sampler=sampler_al,
            num_workers=self.config.DATASET.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=4 if self.config.DATASET.NUM_WORKERS > 0 else None,
        )

    def get_test_loader(self):
        sampler_val = torch.utils.data.RandomSampler(self.test_dataset)

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.DATASET.BATCH_SIZE,
            sampler=sampler_val,
            shuffle=False,
            num_workers=self.config.DATASET.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=4 if self.config.DATASET.NUM_WORKERS > 0 else None,
        )

    def is_valid_cycle(self, cycle_idx):
        return len(self.unlabeled_dataset.get_keys()) > 0 and (cycle_idx < self.num_al_cycles)

    def score_predictions(self, predictions):
        raise NotImplementedError()

    def update(self, cycle_idx):
        raise NotImplementedError()

    def next_cycle(self):
        self.cycle_idx += 1

    def get_training_size(self):
        return len(self.get_train_dataset())

    def get_labeled_size(self):
        return len(self.labeled_dataset.get_keys())

    def get_pseudo_labeled_size(self):
        return len(self.pseudo_labeled_dataset.get_keys())

    def get_unlabeled_size(self):
        return len(self.unlabeled_dataset.get_keys())

    def reset_scores(self):
        self.scores = {}

    def save_scores(self, cycle_idx):
        with open(
            os.path.join(self.config.RUN_DIR, f"unlabeled_scores_cycle_{cycle_idx}.csv"), "w"
        ) as f:
            f.write("key,score\n")
            for key, score in self.scores.items():
                f.write(f"{key},{score}\n")

    def save_labeled_and_unlabeled(self, cycle_idx):
        with open(
            os.path.join(self.config.RUN_DIR, f"unlabeled_cycle_{cycle_idx}.csv"),
            "w",
        ) as f:
            f.write("path,eye_x\n")
            for path, eye_x in self.unlabeled_dataset.get_keys():
                f.write(f"{path},{eye_x}\n")

        with open(
            os.path.join(self.config.RUN_DIR, f"labeled_cycle_{cycle_idx}.csv"),
            "w",
        ) as f:
            f.write("path,eye_x\n")
            for path, eye_x in self.labeled_dataset.get_keys():
                f.write(f"{path},{eye_x}\n")

    def save_pseudo_labeled(self, cycle_idx):
        with open(
            os.path.join(self.config.RUN_DIR, f"pseudo_labeled_cycle_{cycle_idx}.csv"),
            "w",
        ) as f:
            f.write("path,eye_x,gaze_x,gaze_y\n")
            for path, eye_x in self.pseudo_labeled_dataset.get_keys():
                gaze_coords = self.pseudo_labeled_dataset.pseudo_annotations[(path, eye_x)][
                    "gaze_coords"
                ]
                f.write(f"{path},{eye_x},{gaze_coords[0]},{gaze_coords[1]}\n")
