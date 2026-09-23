import torch
import torch.nn.functional as F


def masked_rmse(prediction, target, free_space):
    prediction = prediction.squeeze(1) if prediction.ndim == 4 else prediction
    squared_error = ((prediction - target) * free_space).square()
    return torch.sqrt(squared_error.sum((1, 2)) / free_space.sum((1, 2)).clamp_min(1.0))


def center_of_mass(heatmap):
    heatmap = heatmap.squeeze(1) if heatmap.ndim == 4 else heatmap
    weights = heatmap.clamp_min(0)
    batch, height, width = weights.shape
    rows = torch.arange(height, device=weights.device, dtype=weights.dtype)[None, :, None]
    cols = torch.arange(width, device=weights.device, dtype=weights.dtype)[None, None, :]
    denominator = weights.sum((1, 2))
    row = (weights * rows).sum((1, 2)) / denominator.clamp_min(1e-12)
    col = (weights * cols).sum((1, 2)) / denominator.clamp_min(1e-12)
    fallback = heatmap.flatten(1).argmax(1)
    row = torch.where(denominator > 1e-11, row, (fallback // width).to(row.dtype))
    col = torch.where(denominator > 1e-11, col, (fallback % width).to(col.dtype))
    return torch.stack((row, col), dim=1)


def localization_error(prediction, target_antenna_map):
    prediction_xy = center_of_mass(prediction)
    target = target_antenna_map.squeeze(1) if target_antenna_map.ndim == 4 else target_antenna_map
    index = target.flatten(1).argmax(1)
    width = target.shape[-1]
    target_xy = torch.stack((index // width, index % width), dim=1).to(prediction_xy.dtype)
    return torch.linalg.vector_norm(prediction_xy - target_xy, dim=1)


def masked_ssim(prediction, target, free_space, window_size=11, sigma=1.5):
    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if free_space.ndim == 3:
        free_space = free_space[:, None]
    coords = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    coords -= (window_size - 1) / 2
    kernel_1d = torch.exp(-(coords.square()) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :])[None, None]
    padding = window_size // 2
    mu_x = F.conv2d(prediction, kernel, padding=padding)
    mu_y = F.conv2d(target, kernel, padding=padding)
    var_x = F.conv2d(prediction.square(), kernel, padding=padding) - mu_x.square()
    var_y = F.conv2d(target.square(), kernel, padding=padding) - mu_y.square()
    cov_xy = F.conv2d(prediction * target, kernel, padding=padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return (score * free_space).sum((1, 2, 3)) / free_space.sum((1, 2, 3)).clamp_min(1.0)
