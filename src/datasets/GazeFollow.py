import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

from src.utils.gaze_ops import (
    get_head_mask,
    get_label_map,
)


class GazeFollow(Dataset):
    def __init__(
        self,
        data_dir,
        seed=1,
        input_size=224,
        output_size=64,
        subset="labeled",
        subset_size=1,
        head_bbox_overflow_coeff=0.1,  # Will increase/decrease the bbox of the head by this value (%)
        rgb_transform=None,
        depth_transform=None,
        face_transform=None,
        override_keys_path=None,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.subset = subset
        self.subset_size = subset_size
        self.head_bbox_overflow_coeff = head_bbox_overflow_coeff
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.face_transform = face_transform
        self.override_keys_path = override_keys_path

        assert self.subset in [
            "labeled",
            "pseudo_labeled",
            "unlabeled",
            "test",
        ], f"Invalid subset: {self.subset}"

        # Load all gaze and object annotations
        self._load_gaze_dataset()
        self._load_aux_objects_dataset()

        # If we are splitting the train set into labeled and unlabeled, load the corresponding subset keys
        if (
            self.subset in ["labeled", "unlabeled"]
            and self.subset_size < 1
            and self.subset_size > 0
        ):
            self._load_subset_keys()

        # Store pseudo annotations (i.e. gaze points)
        self.pseudo_annotations = {}
        self.pseudo_gaze_keys = []

    def add_pseudo_annotations(self, key, value):
        raise NotImplementedError("This method should be implemented in 'pseudo' class")

    def get_keys(self):
        if (
            self.subset in ["labeled", "unlabeled"]
            and self.subset_size < 1
            and self.subset_size > 0
        ):
            return self.subset_gaze_keys
        elif self.subset in ["pseudo_labeled"]:
            return self.pseudo_gaze_keys
        else:
            return self.full_gaze_keys

    def _load_gaze_dataset(self):
        labels_path = os.path.join(
            self.data_dir,
            (
                "train_annotations_release.txt"
                if self.subset in ["labeled", "unlabeled", "pseudo_labeled"]
                else "test_annotations_release.txt"
            ),
        )

        column_names = [
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]
        if self.subset in [
            "labeled",
            "unlabeled",
            "pseudo_labeled",
        ]:  # This subsets are from the train_annotations_release.txt
            column_names.append("inout")
        column_names.append("orig_dataset")

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            index_col=False,
        )

        # (-1 is invalid, 0 is out gaze)
        if self.subset in ["labeled", "unlabeled", "pseudo_labeled"]:
            df = df[df["inout"] != -1]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["bbox_x_min"].values,
                    df["bbox_y_min"].values,
                    df["bbox_x_max"].values,
                    df["bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)
        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby(["path", "eye_x"])

        self.full_gaze_annotations = df
        self.full_gaze_keys = list(df.groups.keys())

    def _load_aux_objects_dataset(self):
        labels_path = os.path.join(
            self.data_dir,
            (
                "train_objects.csv"
                if self.subset in ["labeled", "unlabeled", "pseudo_labeled"]
                else "test_objects.csv"
            ),
        )

        column_names = [
            "path",
            "conf",
            "class",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            skiprows=[
                0,
            ],
            index_col=False,
        )

        # Keep only objects with score > min_object_score
        df = df[df["conf"] >= 0.5]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["x_min"].values,
                    df["y_min"].values,
                    df["x_max"].values,
                    df["y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.full_aux_objects_annotations = df
        self.full_aux_objects_keys = list(df.groups.keys())

    def _load_subset_keys(self):
        subset_path = os.path.join(
            self.data_dir, "keys_for_al", f"train_keys_seed_{self.seed}.txt"
        )

        if self.override_keys_path is not None:
            print(f"Overriding {self.subset} keys with {self.override_keys_path}")
            subset_path = self.override_keys_path

        # Check that file exist, otherwise stop
        if not os.path.isfile(subset_path):
            raise FileNotFoundError(f"File {subset_path} not found")

        df = pd.read_csv(subset_path, index_col=False)
        seeded_keys = [tuple(key) for key in df.values.tolist()]

        if self.override_keys_path is None:
            # If train, take the first subset_size perc of the seeded keys. If al, take the last subset_size perc
            n_of_samples_to_keep = int(
                Decimal(self.subset_size * len(seeded_keys)).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )
            if self.subset == "labeled":
                self.subset_gaze_keys = seeded_keys[:n_of_samples_to_keep]
            elif self.subset == "unlabeled":
                self.subset_gaze_keys = seeded_keys[-n_of_samples_to_keep:]
            else:
                raise ValueError(f"Invalid subset: {self.subset}")
        else:
            self.subset_gaze_keys = seeded_keys

    def _get_gaze_coords(self, sample_key, row):
        return [row["gaze_x"], row["gaze_y"]]

    def __len__(self):
        return len(self.get_keys())

    def __getitem__(self, index):
        sample_key = self.get_keys()[index]
        is_labelled = self.subset in ["labeled", "test"]

        # Load scene image
        img_path = sample_key[0]
        image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        img_width, img_height = image.size

        # Load depth image
        depth_path = img_path.replace(".jpg", "_depth.jpg")
        depth = Image.open(os.path.join(self.data_dir, depth_path)).convert("L")

        heads_bbox = []
        eyes_coords = []
        gaze_coords = []
        gaze_inside = []
        for _, row in self.full_gaze_annotations.get_group(sample_key).iterrows():
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]
            gaze_x, gaze_y = self._get_gaze_coords(sample_key, row)
            eye_x = row["eye_x"]
            eye_y = row["eye_y"]

            # Expand head bbox
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

            # All ground truth gaze are stacked up
            heads_bbox.append([x_min, y_min, x_max, y_max])
            eyes_coords.append([eye_x, eye_y])
            gaze_coords.append([gaze_x, gaze_y])

            if self.subset in ["labeled", "unlabeled", "pseudo_labeled"]:
                gaze_inside.append(row["inout"] == 1)
            else:
                gaze_inside.append(True)

        objects_bbox = []
        objects_class = []
        objects_conf = []
        if sample_key[0] in self.full_aux_objects_keys:
            for _, row in self.full_aux_objects_annotations.get_group(sample_key[0]).iterrows():
                x_min = row["x_min"] * img_width
                y_min = row["y_min"] * img_height
                x_max = row["x_max"] * img_width
                y_max = row["y_max"] * img_height
                objects_bbox.append([x_min, y_min, x_max, y_max])
                objects_class.append(row["class"])
                objects_conf.append(row["conf"])

        heads_bbox = torch.Tensor(heads_bbox)
        eyes_coords = torch.FloatTensor(eyes_coords) * torch.FloatTensor([img_width, img_height])
        gaze_coords = torch.FloatTensor(gaze_coords) * torch.FloatTensor([img_width, img_height])
        gaze_inside = torch.FloatTensor(gaze_inside)
        objects_bbox = torch.FloatTensor(objects_bbox)
        objects_class = torch.LongTensor(objects_class)
        objects_conf = torch.FloatTensor(objects_conf)

        # =============================================================================================================
        # Data augmentation (labeled or pseudo-labeled only)
        # =============================================================================================================

        # Jitter (expansion-only) bounding box size
        if self.subset in ["labeled", "pseudo_labeled"] and np.random.random_sample() <= 0.5:
            head_bbox_overflow_coeff = np.random.random_sample() * 0.2
            heads_bbox[:, 0] -= head_bbox_overflow_coeff * abs(heads_bbox[:, 2] - heads_bbox[:, 0])
            heads_bbox[:, 1] -= head_bbox_overflow_coeff * abs(heads_bbox[:, 3] - heads_bbox[:, 1])
            heads_bbox[:, 2] += head_bbox_overflow_coeff * abs(heads_bbox[:, 2] - heads_bbox[:, 0])
            heads_bbox[:, 3] += head_bbox_overflow_coeff * abs(heads_bbox[:, 3] - heads_bbox[:, 1])

        # Random flip
        if (
            self.subset in ["labeled", "pseudo_labeled"]
            and np.random.random_sample() <= 0.5
        ):
            image = F.hflip(image)
            depth = F.hflip(depth)

            # Head bbox
            x_max_2 = img_width - heads_bbox[:, 0]
            x_min_2 = img_width - heads_bbox[:, 2]
            heads_bbox[:, 2] = x_max_2
            heads_bbox[:, 0] = x_min_2

            # Object bbox
            if len(objects_bbox) > 0:
                x_max_2 = img_width - objects_bbox[:, 0]
                x_min_2 = img_width - objects_bbox[:, 2]
                objects_bbox[:, 2] = x_max_2
                objects_bbox[:, 0] = x_min_2

            # Update the bounding boxes
            gaze_coords[:, 0] = img_width - gaze_coords[:, 0]

            # Update the eye coordinates
            eyes_coords[:, 0] = img_width - eyes_coords[:, 0]

        # Random Crop
        if (
            self.subset in ["labeled", "pseudo_labeled"]
            and np.random.random_sample() <= 0.5
        ):
            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            x_coords = torch.cat(
                [
                    gaze_coords[:, 0].flatten(),
                    heads_bbox[:, 0].flatten(),
                    heads_bbox[:, 2].flatten(),
                ]
            )
            y_coords = torch.cat(
                [
                    gaze_coords[:, 1].flatten(),
                    heads_bbox[:, 1].flatten(),
                    heads_bbox[:, 3].flatten(),
                ]
            )
            crop_x_min = x_coords.min().item()
            crop_y_min = y_coords.min().item()
            crop_x_max = x_coords.max().item()
            crop_y_max = y_coords.max().item()

            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = np.random.uniform(0, crop_x_min)
            if crop_y_min >= 0:
                crop_y_min = np.random.uniform(0, crop_y_min)

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min
            crop_height_min = crop_y_max - crop_y_min
            crop_width_max = img_width - crop_x_min
            crop_height_max = img_height - crop_y_min

            # Randomly select a width and a height
            crop_width = np.random.uniform(crop_width_min, crop_width_max)
            crop_height = np.random.uniform(crop_height_min, crop_height_max)

            # Crop it
            image = F.crop(
                image,
                crop_y_min,
                crop_x_min,
                crop_height,
                crop_width,
            )

            # Record the crop's (x, y) offset
            offset_x, offset_y = crop_x_min, crop_y_min

            # Update the bounding boxes
            heads_bbox[:, 0] -= offset_x
            heads_bbox[:, 1] -= offset_y
            heads_bbox[:, 2] -= offset_x
            heads_bbox[:, 3] -= offset_y

            # Update the gaze coordinates
            gaze_coords[:, 0] -= offset_x
            gaze_coords[:, 1] -= offset_y

            # Update the eye coordinates
            eyes_coords[:, 0] -= offset_x
            eyes_coords[:, 1] -= offset_y

            # Remove the object boxes that are outside the crop
            if len(objects_bbox) > 0:
                objects_bbox[:, 0] -= offset_x
                objects_bbox[:, 1] -= offset_y
                objects_bbox[:, 2] -= offset_x
                objects_bbox[:, 3] -= offset_y
                valid_objects_bbox = (
                    (objects_bbox[:, 0] >= 0)
                    & (objects_bbox[:, 1] >= 0)
                    & (objects_bbox[:, 2] <= crop_width)
                    & (objects_bbox[:, 3] <= crop_height)
                )
                objects_bbox = objects_bbox[valid_objects_bbox, :]
                objects_class = objects_class[valid_objects_bbox]
                objects_conf = objects_conf[valid_objects_bbox]

            img_width, img_height = crop_width, crop_height

        # Random color change
        if self.subset in ["labeled", "pseudo_labeled"] and np.random.random_sample() <= 0.5:
            image = F.adjust_brightness(image, brightness_factor=np.random.uniform(0.5, 1.5))
            image = F.adjust_contrast(image, contrast_factor=np.random.uniform(0.5, 1.5))
            image = F.adjust_saturation(image, saturation_factor=np.random.uniform(0, 1.5))

        # =============================================================================================================
        # End of data augmentation (train only)
        # =============================================================================================================

        # Normalize the coords to [0, 1]
        heads_bbox = heads_bbox / torch.FloatTensor([img_width, img_height, img_width, img_height])
        eyes_coords = eyes_coords / torch.FloatTensor([img_width, img_height])
        gaze_coords = gaze_coords / torch.FloatTensor([img_width, img_height])

        # Pad dummy gaze to match size for batch processing
        max_len = 20
        heads_bbox = torch.cat([heads_bbox, torch.full((max_len - len(heads_bbox), 4), -1)], dim=0)
        eyes_coords = torch.cat(
            [eyes_coords, torch.full((max_len - len(eyes_coords), 2), -1)], dim=0
        )
        gaze_coords = torch.cat(
            [gaze_coords, torch.full((max_len - len(gaze_coords), 2), -1)], dim=0
        )
        gaze_inside = (gaze_inside.mean(dim=0) > 0.5).long()

        max_object_len = 300
        objects_bbox = torch.cat(
            [objects_bbox, torch.full((max_object_len - len(objects_bbox), 4), -1)], dim=0
        )
        objects_class = torch.cat(
            [objects_class, torch.full((max_object_len - len(objects_class),), -1)], dim=0
        )
        objects_conf = torch.cat(
            [objects_conf, torch.full((max_object_len - len(objects_conf),), 0)], dim=0
        )

        # Normalize bbox
        objects_bbox = objects_bbox / torch.FloatTensor(
            [img_width, img_height, img_width, img_height]
        )

        # Assert that all bboxes not padding are equal
        assert (
            heads_bbox[heads_bbox[:, 0] != -1, :] == heads_bbox[heads_bbox[:, 0] != -1, :]
        ).all(), "All bboxes should be equal"

        head_bbox = heads_bbox[0, :]
        head_x_min, head_y_min, head_x_max, head_y_max = head_bbox * torch.FloatTensor(
            [img_width, img_height, img_width, img_height]
        )
        head_x_min, head_y_min, head_x_max, head_y_max = (
            head_x_min.item(),
            head_y_min.item(),
            head_x_max.item(),
            head_y_max.item(),
        )
        head_mask = get_head_mask(
            head_x_min,
            head_y_min,
            head_x_max,
            head_y_max,
            img_width,
            img_height,
            resolution=self.input_size,
        )

        # Crop the face
        head = F.crop(
            image,
            head_y_min,
            head_x_min,
            head_y_max - head_y_min,
            head_x_max - head_x_min,
        )

        # Apply transformation to images...
        if self.rgb_transform is not None:
            image = self.rgb_transform(image)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # ... and face
        if self.face_transform is not None:
            head = self.face_transform(head)

        # Generate the heat map used for deconv prediction
        gaze_heatmaps = []
        num_valid = 0
        for gaze_x, gaze_y in gaze_coords:
            if gaze_x < 0:
                continue

            num_valid += 1
            gaze_heatmap = get_label_map(
                torch.zeros(self.output_size, self.output_size),
                [gaze_x * self.output_size, gaze_y * self.output_size],
                3,
                pdf="Gaussian",
            )

            gaze_heatmaps.append(gaze_heatmap)

        gaze_heatmap = torch.stack(gaze_heatmaps, dim=0).mean(dim=0)

        return (
            image,
            depth,
            head,
            head_bbox,
            head_mask,
            gaze_heatmap,
            eyes_coords,
            gaze_coords,
            gaze_inside,
            torch.IntTensor([img_width, img_height]),
            sample_key,
            torch.BoolTensor([is_labelled]).squeeze(),
            objects_bbox,
            objects_class,
            objects_conf,
        )
