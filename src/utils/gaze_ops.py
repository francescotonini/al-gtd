import numpy as np
import torch
from torchmetrics.functional import auroc, average_precision

from src.utils.misc import to_numpy, to_torch


def get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution):
    head_box = (
        np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    )
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)
    head_channel = np.zeros((resolution, resolution), dtype=np.float32)
    head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)

    return head_channel


def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [
        pt[0].round().int().item() - 3 * sigma,
        pt[1].round().int().item() - 3 * sigma,
    ]
    br = [
        pt[0].round().int().item() + 3 * sigma + 1,
        pt[1].round().int().item() + 3 * sigma + 1,
    ]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # Normalize heatmap so it has max value of 1

    return to_torch(img)


def get_onehot_tgt_heatmap(gaze_pts, out_res, device=torch.device("cpu")):
    h, w = out_res
    target_map = torch.zeros((h, w), device=device).long()
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map


def get_heatmap_auc(pred_heatmap, tgt_heatmap):
    return auroc(
        pred_heatmap.flatten(),
        tgt_heatmap.flatten(),
        task="binary",
        num_classes=pred_heatmap.numel(),
    )


def get_heatmap_peak_coords(heatmap):
    np_heatmap = to_numpy(heatmap)

    idx = np.unravel_index(np_heatmap.argmax(), np_heatmap.shape)
    pred_y, pred_x = map(float, idx)

    # Convert to normalized coordinates
    pred_x /= np_heatmap.shape[1]
    pred_y /= np_heatmap.shape[0]

    return torch.tensor([pred_x, pred_y], device=heatmap.device).unsqueeze(0)


def get_l2_dist(p1, p2):
    return torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def get_ap(pred, tgt):
    return average_precision(pred, tgt, task="binary")


def get_angular_error(p1, p2):
    norm_p1 = (p1[:, 0] ** 2 + p1[:, 1] ** 2) ** 0.5
    norm_p2 = (p2[:, 0] ** 2 + p2[:, 1] ** 2) ** 0.5
    cos_sim = (p1[:, 0] * p2[:, 0] + p1[:, 1] * p2[:, 1]) / (norm_p2 * norm_p1 + 1e-6)
    cos_sim = torch.clamp(cos_sim, -1, 1)

    return torch.arccos(cos_sim) * 180 / torch.pi

def points_inside_boxes(orig_points, bboxes):
    points = orig_points.unsqueeze(1).repeat(1, len(bboxes), 1)

    x_min_cond = points[:, :, 0] >= bboxes[:, 0]
    y_min_cond = points[:, :, 1] >= bboxes[:, 1]
    x_max_cond = points[:, :, 0] <= bboxes[:, 2]
    y_max_cond = points[:, :, 1] <= bboxes[:, 3]

    # If point is inside bbox, then sum of above conditions will be 4
    return (x_min_cond.int() + y_min_cond.int() + x_max_cond.int() + y_max_cond.int()) == 4


def get_uncertainty(heatmap, points_to_sample=20, reduction="mean"):
    # Clamp heatmap to 0 and 1
    heatmap = heatmap.clamp(0, 1)

    # Get peak coords and zero-out heatmap
    max_peak_coords = (get_heatmap_peak_coords(heatmap) * 64).squeeze().long()
    max_value = heatmap[max_peak_coords[1], max_peak_coords[0]].clone()
    heatmap[max_peak_coords[1], max_peak_coords[0]] = 0

    # Quantize into 100 bins
    heatmap = torch.round(heatmap * 100) / 100

    other_peaks = [max_value]
    other_coords = [max_peak_coords]
    for _ in range(points_to_sample):
        peak_value = heatmap.max()
        if peak_value == 0:
            break
        other_peaks.append(peak_value)

        # Find coordinates of all the pixels that have the same value as the peak
        peak_coords = (heatmap == peak_value).nonzero()
        # Flip x and y
        peak_coords = peak_coords.flip(1)

        # Find the new peak farthest from the previous peak
        dist = get_l2_dist(max_peak_coords.unsqueeze(0), peak_coords)
        max_dist_idx = dist.argmax()

        # Save the new peak and zero-out the heatmap
        other_coords.append(peak_coords[max_dist_idx])
        heatmap[peak_coords[:, 1], peak_coords[:, 0]] = 0

    other_coords = torch.stack(other_coords)
    other_peaks = torch.stack(other_peaks)

    dists = get_l2_dist(max_peak_coords.unsqueeze(0), other_coords)
    dists = dists / 64 / torch.sqrt(torch.tensor(2.0))

    if reduction == "mean":
        cum_dists = dists.mean() + other_peaks.mean()
    elif reduction == "sum":
        cum_dists = dists.sum() + other_peaks.sum()
    elif reduction == "none":
        return dists, other_peaks

    total_uncertainty = cum_dists
    return total_uncertainty, max_value


def get_heatmap_peak_coords_batched(heatmap_batch):
    _, max_indices = torch.max(heatmap_batch.view(heatmap_batch.size(0), -1), dim=1)
    max_indices = max_indices.view(-1, 1)

    # Convert the flattened indices to 2D coordinates
    max_y, max_x = max_indices // heatmap_batch.size(-1), max_indices % heatmap_batch.size(-1)
    pred_x = max_x / (heatmap_batch.size(-1))
    pred_y = max_y / (heatmap_batch.size(-2))

    # Stack the coordinates along the last dimension
    peak_coords = torch.cat((pred_x, pred_y), dim=1)
    return peak_coords
