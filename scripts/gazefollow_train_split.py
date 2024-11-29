import argparse
import os
import random

import numpy as np
import pandas as pd
import torch


def get_keys(data_dir):
    labels_path = os.path.join(data_dir, "train_annotations_release.txt")

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
        "inout",
        "orig_dataset",
    ]

    # Load annotations
    df = pd.read_csv(
        labels_path,
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )
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

    return list(df.groups.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2, 5, 11, 23, 31, 47, 59, 61, 73, 89]
    )
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"Seed: {seed}")

        # Seed everything
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Set output dir
        output_dir = os.path.join(args.dataset_dir, "keys_for_al")

        # Create output dir if missing
        os.makedirs(output_dir, exist_ok=True)

        # Get and shuffle keys
        keys = get_keys(args.dataset_dir)
        np.random.shuffle(keys)

        with open(os.path.join(output_dir, f"train_keys_seed_{seed}.txt"), "w") as f:
            for key in keys:
                f.write(f"{key[0]},{key[1]}\n")

    print("Done!")
