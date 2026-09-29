"""Plateau-aware decoding and cardinality-normalized localization metrics."""

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import linear_sum_assignment


def decode(heatmap, threshold=0.25, nms_size=11):
    peaks = (heatmap == maximum_filter(heatmap, size=nms_size, mode="constant")) & (
        heatmap > threshold
    )
    regions, count = label(peaks)
    points, scores = [], []
    for i in range(1, count + 1):
        region = regions == i
        y, x = np.where(region)
        points.append([float(x.mean()), float(y.mean())])
        scores.append(float(heatmap[region].max()))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)[np.argsort(scores)[::-1]]


def localization_metrics(predicted, target, cutoff=20.0, order=2):
    n, m = len(predicted), len(target)
    mle = None
    if min(n, m):
        distance = np.linalg.norm(predicted[:, None, :] - target[None, :, :], axis=-1)
        cost = np.minimum(distance, cutoff) ** order
        rows, cols = linear_sum_assignment(cost)
        ospa = (
            (cost[rows, cols].sum() + cutoff**order * abs(n - m)) / max(n, m)
        ) ** (1 / order)
        rows, cols = linear_sum_assignment(distance)
        mle = float(distance[rows, cols].mean())
    else:
        ospa = cutoff if max(n, m) else 0.0
    miss, fa = max(m - n, 0), max(n - m, 0)
    return {
        "ospa": float(ospa),
        "mle": mle,
        "miss_count": miss,
        "fa_count": fa,
        "mdr": miss / m if m else None,
        "far": fa / m if m else None,
        "num_gt": m,
        "num_pred": n,
    }


def postprocess(task):
    heatmap, target, threshold, nms_size, cutoff, order = task
    points = decode(heatmap, threshold, nms_size)
    return points, localization_metrics(points, target, cutoff, order)
