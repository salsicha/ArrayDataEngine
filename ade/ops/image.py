from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray, min_value=None, max_value=None, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(image)
    min_value = arr.min() if min_value is None else min_value
    max_value = arr.max() if max_value is None else max_value
    span = max_value - min_value
    if span == 0:
        return np.zeros(arr.shape, dtype=dtype)
    return ((arr - min_value) / span).astype(dtype)


def normalize_images(
    images: np.ndarray,
    min_value=None,
    max_value=None,
    dtype=np.float32,
    per_image: bool = False,
) -> np.ndarray:
    """Normalize an image/depth sequence to `[0, 1]`."""

    arr = _as_image_sequence(images)
    if not per_image:
        return normalize_image(arr, min_value=min_value, max_value=max_value, dtype=dtype)

    values = arr.astype(np.float64, copy=False)
    axes = tuple(range(1, values.ndim))
    mins = values.min(axis=axes, keepdims=True) if min_value is None else np.asarray(min_value, dtype=np.float64)
    maxs = values.max(axis=axes, keepdims=True) if max_value is None else np.asarray(max_value, dtype=np.float64)
    span = maxs - mins
    normalized = np.divide(
        values - mins,
        span,
        out=np.zeros_like(values, dtype=np.float64),
        where=span != 0,
    )
    return normalized.astype(dtype)


def pad_image(image: np.ndarray, pad_width, value=0) -> np.ndarray:
    if np.isscalar(pad_width):
        pad = ((int(pad_width), int(pad_width)), (int(pad_width), int(pad_width)))
    elif len(pad_width) == 2 and all(np.isscalar(v) for v in pad_width):
        pad = ((int(pad_width[0]), int(pad_width[0])), (int(pad_width[1]), int(pad_width[1])))
    else:
        pad = pad_width
    if np.asarray(image).ndim == 3 and len(pad) == 2:
        pad = tuple(pad) + ((0, 0),)
    return np.pad(image, pad, mode="constant", constant_values=value)


def pad_images(images: np.ndarray, pad_width, value=0) -> np.ndarray:
    """Pad the spatial dimensions of an image/depth sequence."""

    arr = _as_image_sequence(images)
    spatial_pad = _normalize_spatial_pad(pad_width)
    pad = ((0, 0), *spatial_pad)
    if arr.ndim == 4:
        pad = (*pad, (0, 0))
    return np.pad(arr, pad, mode="constant", constant_values=value)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("shape must contain positive height and width")
    row_idx = np.linspace(0, arr.shape[0] - 1, out_h).round().astype(int)
    col_idx = np.linspace(0, arr.shape[1] - 1, out_w).round().astype(int)
    return arr[row_idx][:, col_idx]


def resize_images_nearest(images: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an image/depth sequence with nearest-neighbor sampling."""

    arr = _as_image_sequence(images)
    out_h, out_w = _validate_shape(shape)
    row_idx = np.linspace(0, arr.shape[1] - 1, out_h).round().astype(int)
    col_idx = np.linspace(0, arr.shape[2] - 1, out_w).round().astype(int)
    return arr[:, row_idx][:, :, col_idx]


def resize_images(images: np.ndarray, shape: tuple[int, int], method: str = "nearest") -> np.ndarray:
    """Resize an image/depth sequence."""

    if method != "nearest":
        raise ValueError("only nearest resize is currently supported")
    return resize_images_nearest(images, shape)


def crop_image(image: np.ndarray, row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
    """Crop a single image or depth map."""

    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    return arr[row_start:row_stop, col_start:col_stop, ...].copy()


def crop_images(images: np.ndarray, row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
    """Crop the spatial dimensions of an image/depth sequence."""

    arr = _as_image_sequence(images)
    return arr[:, row_start:row_stop, col_start:col_stop, ...].copy()


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        raise ValueError("image must have shape (..., H, W, 3+)")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return arr[..., :3] @ weights


def convert_color(images: np.ndarray, mode: str, alpha=None) -> np.ndarray:
    """Convert image or image-sequence color layout."""

    arr = np.asarray(images)
    mode = mode.lower()
    if mode in {"rgb_to_gray", "bgr_to_gray"}:
        source = arr
        if mode == "bgr_to_gray":
            _require_channels(arr, 3)
            source = arr.copy()
            source[..., :3] = source[..., :3][..., ::-1]
        return rgb_to_gray(source)
    if mode in {"rgb_to_bgr", "bgr_to_rgb"}:
        _require_channels(arr, 3)
        result = arr.copy()
        result[..., :3] = result[..., :3][..., ::-1]
        return result
    if mode == "rgba_to_rgb":
        _require_channels(arr, 4)
        return arr[..., :3].copy()
    if mode == "rgb_to_rgba":
        _require_channels(arr, 3)
        alpha_value = _default_alpha(arr.dtype) if alpha is None else alpha
        alpha_channel = np.full((*arr.shape[:-1], 1), alpha_value, dtype=arr.dtype)
        return np.concatenate((arr[..., :3], alpha_channel), axis=-1)
    if mode == "gray_to_rgb":
        gray = _as_gray_image_or_sequence(arr)
        return np.repeat(gray[..., None], 3, axis=-1)
    raise ValueError(
        "mode must be one of rgb_to_gray, bgr_to_gray, rgb_to_bgr, bgr_to_rgb, "
        "rgba_to_rgb, rgb_to_rgba, or gray_to_rgb"
    )


def convert_image_dtype(image: np.ndarray, dtype, scale: bool = True, clip: bool = True) -> np.ndarray:
    """Convert image/depth arrays between common numeric dtypes."""

    arr = np.asarray(image)
    target = np.dtype(dtype)
    if not scale:
        return arr.astype(target)
    if arr.dtype == target:
        return arr.copy()

    if np.issubdtype(target, np.floating):
        converted = arr.astype(np.float64)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            if info.min < 0:
                converted = (converted - info.min) / (info.max - info.min)
            elif info.max > 0:
                converted = converted / info.max
        return converted.astype(target)

    if not np.issubdtype(target, np.integer):
        return arr.astype(target)

    target_info = np.iinfo(target)
    converted = arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.floating):
        if clip:
            converted = np.clip(converted, 0.0, 1.0)
        converted = converted * target_info.max
    elif np.issubdtype(arr.dtype, np.integer):
        source_info = np.iinfo(arr.dtype)
        if source_info.min < 0:
            converted = (converted - source_info.min) / (source_info.max - source_info.min)
            converted = converted * (target_info.max - target_info.min) + target_info.min
        elif source_info.max != target_info.max:
            converted = converted / source_info.max * target_info.max

    if clip:
        converted = np.clip(converted, target_info.min, target_info.max)
    return np.rint(converted).astype(target)


def image_mask(
    image: np.ndarray,
    min_value=None,
    max_value=None,
    inclusive: bool = True,
    finite: bool = True,
) -> np.ndarray:
    """Create a boolean mask from value bounds."""

    arr = np.asarray(image)
    mask = np.ones(arr.shape, dtype=bool)
    if finite:
        mask &= np.isfinite(arr)
    if min_value is not None:
        mask &= arr >= min_value if inclusive else arr > min_value
    if max_value is not None:
        mask &= arr <= max_value if inclusive else arr < max_value
    return mask


def apply_image_mask(image: np.ndarray, mask: np.ndarray, fill_value=0) -> np.ndarray:
    """Apply a boolean mask to an image or image sequence."""

    arr = np.asarray(image)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != arr.shape:
        if keep.shape == arr.shape[: keep.ndim]:
            keep = keep.reshape((*keep.shape, *([1] * (arr.ndim - keep.ndim))))
        else:
            raise ValueError("mask must match the image shape or leading spatial dimensions")
    return np.where(keep, arr, fill_value)


def threshold_image(
    image: np.ndarray,
    threshold,
    above: bool = True,
    inclusive: bool = True,
    high=True,
    low=False,
    dtype=None,
) -> np.ndarray:
    """Threshold an image into a binary or two-valued output array."""

    arr = np.asarray(image)
    if above:
        mask = arr >= threshold if inclusive else arr > threshold
    else:
        mask = arr <= threshold if inclusive else arr < threshold
    result = np.where(mask, high, low)
    if dtype is None:
        dtype = np.result_type(high, low)
    return result.astype(dtype)


def dilate_mask(mask: np.ndarray, size=3, iterations: int = 1, spatial_axes=None) -> np.ndarray:
    """Dilate a boolean mask over spatial dimensions."""

    return _morphology(mask, size=size, iterations=iterations, operation="dilate", spatial_axes=spatial_axes)


def erode_mask(mask: np.ndarray, size=3, iterations: int = 1, spatial_axes=None) -> np.ndarray:
    """Erode a boolean mask over spatial dimensions."""

    return _morphology(mask, size=size, iterations=iterations, operation="erode", spatial_axes=spatial_axes)


def open_mask(mask: np.ndarray, size=3, iterations: int = 1, spatial_axes=None) -> np.ndarray:
    """Remove small mask islands with erosion followed by dilation."""

    eroded = erode_mask(mask, size=size, iterations=iterations, spatial_axes=spatial_axes)
    return dilate_mask(eroded, size=size, iterations=iterations, spatial_axes=spatial_axes)


def close_mask(mask: np.ndarray, size=3, iterations: int = 1, spatial_axes=None) -> np.ndarray:
    """Fill small mask holes with dilation followed by erosion."""

    dilated = dilate_mask(mask, size=size, iterations=iterations, spatial_axes=spatial_axes)
    return erode_mask(dilated, size=size, iterations=iterations, spatial_axes=spatial_axes)


def image_gradients(image: np.ndarray, method: str = "sobel", spatial_axes=None) -> dict[str, np.ndarray]:
    """Compute spatial image gradients and magnitude."""

    arr = _gradient_input(image)
    axes = _infer_spatial_axes(arr, spatial_axes)
    method = method.lower()
    if method == "central":
        dy, dx = np.gradient(arr.astype(np.float64, copy=False), axis=axes)
    elif method == "sobel":
        dx = _convolve_spatial(arr, np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 8.0, axes)
        dy = _convolve_spatial(arr, np.array([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]) / 8.0, axes)
    else:
        raise ValueError("method must be 'sobel' or 'central'")

    return {
        "dy": dy,
        "dx": dx,
        "magnitude": np.hypot(dy, dx),
    }


def image_pyramid(image: np.ndarray, levels: int, downscale: float = 2.0, method: str = "nearest") -> list[np.ndarray]:
    """Build an image pyramid including the original image as level 0."""

    if method != "nearest":
        raise ValueError("only nearest pyramid downsampling is currently supported")
    levels = int(levels)
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if downscale <= 1.0:
        raise ValueError("downscale must be greater than 1")

    current = np.asarray(image).copy()
    pyramid = [current]
    for _ in range(1, levels):
        axes = _infer_spatial_axes(current, None)
        out_h = max(1, int(np.ceil(current.shape[axes[0]] / downscale)))
        out_w = max(1, int(np.ceil(current.shape[axes[1]] / downscale)))
        current = _resize_spatial_nearest(current, (out_h, out_w), axes)
        pyramid.append(current)
    return pyramid


def local_statistics(
    image: np.ndarray,
    size=3,
    statistics=("mean", "std"),
    spatial_axes=None,
) -> dict[str, np.ndarray]:
    """Compute local window statistics over spatial dimensions."""

    arr = np.asarray(image)
    axes = _infer_spatial_axes(arr, spatial_axes)
    windows = _spatial_windows(arr, size=size, spatial_axes=axes, pad_mode="edge")
    result = {}
    for statistic in statistics:
        name = statistic.lower()
        if name == "mean":
            result["mean"] = windows.mean(axis=(-2, -1))
        elif name == "std":
            result["std"] = windows.astype(np.float64, copy=False).std(axis=(-2, -1))
        elif name == "min":
            result["min"] = windows.min(axis=(-2, -1))
        elif name == "max":
            result["max"] = windows.max(axis=(-2, -1))
        else:
            raise ValueError("statistics entries must be mean, std, min, or max")
    return result


def local_mean(image: np.ndarray, size=3, spatial_axes=None) -> np.ndarray:
    return local_statistics(image, size=size, statistics=("mean",), spatial_axes=spatial_axes)["mean"]


def local_std(image: np.ndarray, size=3, spatial_axes=None) -> np.ndarray:
    return local_statistics(image, size=size, statistics=("std",), spatial_axes=spatial_axes)["std"]


def estimate_image_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Estimate integer ``(row_shift, col_shift)`` from reference to moving image."""

    ref, mov = _motion_pair(reference, moving)
    return _phase_correlation_shift(ref, mov, max_shift=max_shift)


def frame_optical_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: int | tuple[int, int] | None = None,
    block_size: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Estimate a dense frame-to-frame flow field with ``(dy, dx)`` channels."""

    ref, mov = _motion_pair(reference, moving)
    height, width = ref.shape
    flow = np.empty((height, width, 2), dtype=np.float64)
    if block_size is None:
        shift = _phase_correlation_shift(ref, mov, max_shift=max_shift)
        flow[..., 0] = shift[0]
        flow[..., 1] = shift[1]
        return flow

    block_h, block_w = _kernel_shape(block_size)
    for row_start in range(0, height, block_h):
        row_stop = min(row_start + block_h, height)
        for col_start in range(0, width, block_w):
            col_stop = min(col_start + block_w, width)
            shift = _phase_correlation_shift(
                ref[row_start:row_stop, col_start:col_stop],
                mov[row_start:row_stop, col_start:col_stop],
                max_shift=max_shift,
            )
            flow[row_start:row_stop, col_start:col_stop, 0] = shift[0]
            flow[row_start:row_stop, col_start:col_stop, 1] = shift[1]
    return flow


def iter_frame_optical_flow(
    images,
    max_shift: int | tuple[int, int] | None = None,
    block_size: int | tuple[int, int] | None = None,
):
    """Yield optical flow for adjacent frames without materializing a flow stack."""

    iterator = iter(_iter_image_frames(images))
    try:
        previous = next(iterator)
    except StopIteration:
        return

    for current in iterator:
        yield frame_optical_flow(previous, current, max_shift=max_shift, block_size=block_size)
        previous = current


def frame_to_frame_optical_flow(
    images: np.ndarray,
    max_shift: int | tuple[int, int] | None = None,
    block_size: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Collect adjacent-frame optical flow fields into a ``(N - 1, H, W, 2)`` stack."""

    arr = _as_image_sequence(images)
    height, width = arr.shape[1], arr.shape[2]
    flows = list(iter_frame_optical_flow(arr, max_shift=max_shift, block_size=block_size))
    if not flows:
        return np.empty((0, height, width, 2), dtype=np.float64)
    return np.stack(flows, axis=0)


def translate_image(image: np.ndarray, shift, fill_value=0) -> np.ndarray:
    """Translate an image by integer ``(row_shift, col_shift)`` pixels."""

    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    row_shift, col_shift = _integer_shift(shift)
    result = np.full(arr.shape, fill_value, dtype=arr.dtype)
    height, width = arr.shape[:2]

    src_row_start = max(0, -row_shift)
    src_row_stop = height - max(0, row_shift)
    dst_row_start = max(0, row_shift)
    dst_row_stop = height - max(0, -row_shift)
    src_col_start = max(0, -col_shift)
    src_col_stop = width - max(0, col_shift)
    dst_col_start = max(0, col_shift)
    dst_col_stop = width - max(0, -col_shift)

    if src_row_start >= src_row_stop or src_col_start >= src_col_stop:
        return result
    result[dst_row_start:dst_row_stop, dst_col_start:dst_col_stop, ...] = arr[
        src_row_start:src_row_stop,
        src_col_start:src_col_stop,
        ...,
    ]
    return result


def align_image(
    moving: np.ndarray,
    reference: np.ndarray | None = None,
    shift=None,
    max_shift: int | tuple[int, int] | None = None,
    fill_value=0,
    return_shift: bool = False,
):
    """Align one moving image to a reference using estimated translation."""

    if shift is None:
        if reference is None:
            raise ValueError("reference is required when shift is not provided")
        shift = estimate_image_shift(reference, moving, max_shift=max_shift)
    shift = np.asarray(shift, dtype=np.float64)
    aligned = translate_image(moving, -shift, fill_value=fill_value)
    if return_shift:
        return aligned, shift
    return aligned


def iter_aligned_images(
    images,
    max_shift: int | tuple[int, int] | None = None,
    fill_value=0,
    incremental: bool = True,
    return_shifts: bool = False,
):
    """Yield images aligned to the first frame without collecting the sequence."""

    iterator = iter(_iter_image_frames(images))
    try:
        first = next(iterator)
    except StopIteration:
        return

    cumulative = np.zeros(2, dtype=np.float64)
    previous = first
    yield (first.copy(), cumulative.copy()) if return_shifts else first.copy()

    for current in iterator:
        if incremental:
            shift = estimate_image_shift(previous, current, max_shift=max_shift)
            cumulative = cumulative + shift
        else:
            cumulative = estimate_image_shift(first, current, max_shift=max_shift)
        aligned = translate_image(current, -cumulative, fill_value=fill_value)
        yield (aligned, cumulative.copy()) if return_shifts else aligned
        previous = current


def align_images(
    images: np.ndarray,
    max_shift: int | tuple[int, int] | None = None,
    fill_value=0,
    incremental: bool = True,
    return_shifts: bool = False,
):
    """Collect a sequence aligned to the first frame."""

    if return_shifts:
        pairs = list(iter_aligned_images(
            images,
            max_shift=max_shift,
            fill_value=fill_value,
            incremental=incremental,
            return_shifts=True,
        ))
        if not pairs:
            arr = _as_image_sequence(images)
            return np.empty_like(arr), np.empty((0, 2), dtype=np.float64)
        aligned, shifts = zip(*pairs)
        return np.stack(aligned, axis=0), np.stack(shifts, axis=0)

    aligned = list(iter_aligned_images(
        images,
        max_shift=max_shift,
        fill_value=fill_value,
        incremental=incremental,
    ))
    if not aligned:
        return np.empty_like(_as_image_sequence(images))
    return np.stack(aligned, axis=0)


def iter_motion_compensated_windows(
    images,
    window_size: int,
    max_shift: int | tuple[int, int] | None = None,
    fill_value=0,
    min_periods: int = 1,
    return_shifts: bool = False,
):
    """Yield trailing rolling windows aligned to each window's newest frame."""

    from collections import deque

    window_size = int(window_size)
    min_periods = int(min_periods)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if min_periods < 1 or min_periods > window_size:
        raise ValueError("min_periods must be in [1, window_size]")

    window = deque(maxlen=window_size)
    for frame in _iter_image_frames(images):
        window.append(frame)
        if len(window) < min_periods:
            continue

        reference = window[-1]
        aligned = []
        shifts = []
        for candidate in window:
            shift = estimate_image_shift(reference, candidate, max_shift=max_shift)
            aligned.append(translate_image(candidate, -shift, fill_value=fill_value))
            shifts.append(shift)
        aligned_stack = np.stack(aligned, axis=0)
        shift_stack = np.stack(shifts, axis=0)
        yield (aligned_stack, shift_stack) if return_shifts else aligned_stack


def motion_compensated_rolling_windows(
    images,
    window_size: int,
    max_shift: int | tuple[int, int] | None = None,
    fill_value=0,
    min_periods: int = 1,
    return_shifts: bool = False,
):
    """Return a lazy iterator of trailing motion-compensated image windows."""

    return iter_motion_compensated_windows(
        images,
        window_size=window_size,
        max_shift=max_shift,
        fill_value=fill_value,
        min_periods=min_periods,
        return_shifts=return_shifts,
    )


def valid_depth_mask(depth: np.ndarray, min_depth: float = 0.0, max_depth: float | None = None) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float64)
    mask = np.isfinite(arr) & (arr > min_depth)
    if max_depth is not None:
        mask &= arr <= max_depth
    return mask


def depth_to_points(
    depth: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    fx, fy, cx, cy = _depth_intrinsics(fx, fy, cx, cy, camera_matrix)
    z, valid = _valid_depth_values(depth, scale=scale, mask=mask)
    rows, cols = np.nonzero(valid)
    z_valid = z[rows, cols]
    x = (cols - cx) * z_valid / fx
    y = (rows - cy) * z_valid / fy
    return np.column_stack((x, y, z_valid))


def depth_to_point_grid(
    depth: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject a depth image into an organized ``(H, W, 3)`` point grid."""

    fx, fy, cx, cy = _depth_intrinsics(fx, fy, cx, cy, camera_matrix)
    z, valid = _valid_depth_values(depth, scale=scale, mask=mask)
    rows, cols = np.indices(z.shape, dtype=np.float64)
    points = np.empty((*z.shape, 3), dtype=np.float64)
    points[..., 0] = (cols - cx) * z / fx
    points[..., 1] = (rows - cy) * z / fy
    points[..., 2] = z
    points[~valid] = np.nan
    return points


def depth_to_normals(
    depth: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    orient_toward_camera: bool = True,
) -> np.ndarray:
    """Estimate an organized normal map from a depth image.

    Interior pixels use central differences in backprojected camera space.
    Pixels without valid neighbors are set to ``nan``.
    """

    points = depth_to_point_grid(
        depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scale=scale,
        mask=mask,
        camera_matrix=camera_matrix,
    )
    normals = np.full_like(points, np.nan)
    if points.shape[0] < 3 or points.shape[1] < 3:
        return normals

    dx = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    dy = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    interior = np.cross(dx, dy)
    lengths = np.linalg.norm(interior, axis=-1)
    center_valid = np.isfinite(points[1:-1, 1:-1, :]).all(axis=-1)
    valid = center_valid & np.isfinite(interior).all(axis=-1) & (lengths > 0)
    interior[valid] = interior[valid] / lengths[valid, None]
    interior[~valid] = np.nan

    if orient_toward_camera:
        to_camera = -points[1:-1, 1:-1, :]
        facing = np.einsum("...i,...i->...", interior, to_camera)
        flip = valid & (facing < 0)
        interior[flip] *= -1.0

    normals[1:-1, 1:-1, :] = interior
    return normals


def iter_rgbd_frame_points(
    depth_images: np.ndarray,
    color_images: np.ndarray | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    masks: np.ndarray | None = None,
    transforms: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
):
    """Yield one backprojected RGB-D point cloud per frame."""

    fx, fy, cx, cy = _depth_intrinsics(fx, fy, cx, cy, camera_matrix)
    depths = _as_depth_sequence(depth_images)
    colors = _as_rgbd_color_sequence(color_images, depths.shape) if color_images is not None else None
    mask_sequence = _as_optional_mask_sequence(masks, depths.shape)
    transform_sequence = _as_optional_transform_sequence(transforms, depths.shape[0])
    color_columns = 0 if colors is None else colors.shape[-1]

    for index, depth_frame in enumerate(depths):
        yield _rgbd_frame_to_points(
            depth_frame,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            scale=scale,
            mask=None if mask_sequence is None else mask_sequence[index],
            color=None if colors is None else colors[index],
            color_columns=color_columns,
            transform=None if transform_sequence is None else transform_sequence[index],
        )


def fuse_rgbd_frames(
    depth_images: np.ndarray,
    color_images: np.ndarray | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    masks: np.ndarray | None = None,
    transforms: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse one or more aligned RGB-D frames into a single point cloud.

    The output contains ``xyz`` columns and, when colors are provided, the
    color channels sampled at each valid depth pixel.
    """

    depths = _as_depth_sequence(depth_images)
    colors = _as_rgbd_color_sequence(color_images, depths.shape) if color_images is not None else None
    columns = 3 if colors is None else 3 + colors.shape[-1]
    chunks = list(iter_rgbd_frame_points(
        depths,
        color_images=colors,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scale=scale,
        masks=masks,
        transforms=transforms,
        camera_matrix=camera_matrix,
    ))
    if not chunks:
        return np.empty((0, columns), dtype=np.float64)
    return np.vstack(chunks)


def _as_image_sequence(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim not in (3, 4):
        raise ValueError("images must have shape (N, H, W) or (N, H, W, C)")
    return arr


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    out_h, out_w = (int(shape[0]), int(shape[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError("shape must contain positive height and width")
    return out_h, out_w


def _normalize_spatial_pad(pad_width):
    if np.isscalar(pad_width):
        value = int(pad_width)
        return ((value, value), (value, value))
    if len(pad_width) == 2 and all(np.isscalar(v) for v in pad_width):
        return ((int(pad_width[0]), int(pad_width[0])), (int(pad_width[1]), int(pad_width[1])))
    if len(pad_width) == 2:
        return tuple((int(pair[0]), int(pair[1])) for pair in pad_width)
    raise ValueError("pad_width must be scalar, (rows, cols), or ((top, bottom), (left, right))")


def _require_channels(arr: np.ndarray, count: int) -> None:
    if arr.ndim < 3 or arr.shape[-1] < count:
        raise ValueError(f"image must have at least {count} channels on the last axis")


def _as_gray_image_or_sequence(arr: np.ndarray) -> np.ndarray:
    if arr.ndim in (2, 3):
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError("gray image data must have shape (H, W), (N, H, W), or a singleton channel")


def _default_alpha(dtype) -> float | int:
    if np.issubdtype(dtype, np.floating):
        return 1.0
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1


def _infer_spatial_axes(arr: np.ndarray, spatial_axes) -> tuple[int, int]:
    if spatial_axes is not None:
        axes = tuple(int(axis) for axis in spatial_axes)
        if len(axes) != 2:
            raise ValueError("spatial_axes must contain two axes")
        axes = tuple(axis + arr.ndim if axis < 0 else axis for axis in axes)
        if any(axis < 0 or axis >= arr.ndim for axis in axes) or axes[0] == axes[1]:
            raise ValueError("spatial_axes must refer to two distinct axes")
        return axes

    if arr.ndim == 2:
        return (0, 1)
    if arr.ndim == 3:
        return (0, 1) if arr.shape[-1] in (1, 3, 4) else (1, 2)
    if arr.ndim == 4:
        return (1, 2)
    raise ValueError("image data must have shape (H, W), (H, W, C), (N, H, W), or (N, H, W, C)")


def _kernel_shape(size) -> tuple[int, int]:
    if np.isscalar(size):
        height = width = int(size)
    else:
        height, width = (int(size[0]), int(size[1]))
    if height < 1 or width < 1:
        raise ValueError("size must contain positive dimensions")
    return height, width


def _spatial_windows(image: np.ndarray, size, spatial_axes, pad_mode: str = "edge", constant_values=False) -> np.ndarray:
    arr = np.asarray(image)
    axes = _infer_spatial_axes(arr, spatial_axes)
    height, width = _kernel_shape(size)
    pad = [(0, 0)] * arr.ndim
    pad[axes[0]] = (height // 2, height - 1 - height // 2)
    pad[axes[1]] = (width // 2, width - 1 - width // 2)
    if pad_mode == "constant":
        padded = np.pad(arr, pad, mode=pad_mode, constant_values=constant_values)
    else:
        padded = np.pad(arr, pad, mode=pad_mode)
    return np.lib.stride_tricks.sliding_window_view(padded, (height, width), axis=axes)


def _morphology(mask: np.ndarray, size, iterations: int, operation: str, spatial_axes=None) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if iterations < 1:
        return result.copy()
    for _ in range(int(iterations)):
        windows = _spatial_windows(
            result,
            size=size,
            spatial_axes=spatial_axes,
            pad_mode="constant",
            constant_values=False,
        )
        if operation == "dilate":
            result = windows.any(axis=(-2, -1))
        elif operation == "erode":
            result = windows.all(axis=(-2, -1))
        else:
            raise ValueError("operation must be 'dilate' or 'erode'")
    return result


def _gradient_input(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim >= 3 and arr.shape[-1] >= 3 and arr.shape[-1] <= 4:
        return rgb_to_gray(arr)
    return arr.astype(np.float64, copy=False)


def _convolve_spatial(image: np.ndarray, kernel: np.ndarray, spatial_axes) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    windows = _spatial_windows(arr, size=kernel.shape, spatial_axes=spatial_axes, pad_mode="edge")
    return np.einsum("...ij,ij->...", windows, kernel)


def _resize_spatial_nearest(image: np.ndarray, shape: tuple[int, int], spatial_axes: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    out_h, out_w = _validate_shape(shape)
    row_idx = np.linspace(0, arr.shape[spatial_axes[0]] - 1, out_h).round().astype(int)
    col_idx = np.linspace(0, arr.shape[spatial_axes[1]] - 1, out_w).round().astype(int)
    result = np.take(arr, row_idx, axis=spatial_axes[0])
    return np.take(result, col_idx, axis=spatial_axes[1])


def _iter_image_frames(images):
    if isinstance(images, np.ndarray):
        arr = _as_image_sequence(images)
        for frame in arr:
            yield frame
        return

    try:
        arr = np.asarray(images)
    except ValueError:
        arr = None
    if arr is not None and arr.dtype != object and arr.ndim in (3, 4):
        for frame in arr:
            yield frame
        return

    for frame in images:
        arr_frame = np.asarray(frame)
        if arr_frame.ndim not in (2, 3):
            raise ValueError("each image frame must have shape (H, W) or (H, W, C)")
        yield arr_frame


def _motion_pair(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = _motion_image(reference)
    mov = _motion_image(moving)
    if ref.shape != mov.shape:
        raise ValueError("reference and moving images must have matching spatial shape")
    return ref, mov


def _motion_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = rgb_to_gray(arr)
    if arr.ndim != 2:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    arr = arr.astype(np.float64, copy=False)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: int | tuple[int, int] | None = None,
) -> np.ndarray:
    if reference.size == 0:
        return np.zeros(2, dtype=np.float64)

    ref = reference.astype(np.float64, copy=False) - float(np.mean(reference))
    mov = moving.astype(np.float64, copy=False) - float(np.mean(moving))
    if not np.any(ref) or not np.any(mov):
        return np.zeros(2, dtype=np.float64)

    cross_power = np.fft.fft2(mov) * np.conj(np.fft.fft2(ref))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(cross_power, magnitude, out=np.zeros_like(cross_power), where=magnitude > 1e-12)
    correlation = np.fft.ifft2(cross_power).real
    peak = _bounded_correlation_peak(correlation, max_shift=max_shift)
    return _peak_to_shift(peak, correlation.shape)


def _bounded_correlation_peak(correlation: np.ndarray, max_shift: int | tuple[int, int] | None) -> tuple[int, int]:
    if max_shift is None:
        return tuple(int(index) for index in np.unravel_index(np.argmax(correlation), correlation.shape))

    max_row, max_col = _normalize_max_shift(max_shift)
    height, width = correlation.shape
    max_row = min(max_row, height // 2)
    max_col = min(max_col, width // 2)
    rows = _shift_candidate_indices(height, max_row)
    cols = _shift_candidate_indices(width, max_col)
    bounded = correlation[np.ix_(rows, cols)]
    local = np.unravel_index(np.argmax(bounded), bounded.shape)
    return int(rows[local[0]]), int(cols[local[1]])


def _shift_candidate_indices(size: int, max_shift: int) -> np.ndarray:
    if max_shift <= 0:
        return np.array([0], dtype=int)
    positive = np.arange(0, max_shift + 1, dtype=int)
    negative = np.arange(size - max_shift, size, dtype=int)
    return np.unique(np.concatenate((positive, negative)))


def _peak_to_shift(peak: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    shift = np.array(peak, dtype=np.float64)
    for axis, size in enumerate(shape):
        if shift[axis] > size // 2:
            shift[axis] -= size
    return shift


def _normalize_max_shift(max_shift: int | tuple[int, int]) -> tuple[int, int]:
    if np.isscalar(max_shift):
        row = col = int(max_shift)
    else:
        row, col = int(max_shift[0]), int(max_shift[1])
    if row < 0 or col < 0:
        raise ValueError("max_shift must be non-negative")
    return row, col


def _integer_shift(shift) -> tuple[int, int]:
    arr = np.asarray(shift, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("shift must contain row and column values")
    return int(np.rint(arr[0])), int(np.rint(arr[1]))


def _depth_intrinsics(
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    camera_matrix: np.ndarray | None,
) -> tuple[float, float, float, float]:
    if camera_matrix is not None:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError("fx, fy, cx, and cy are required unless camera_matrix is provided")
    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)
    if fx == 0 or fy == 0:
        raise ValueError("fx and fy must be non-zero")
    return fx, fy, cx, cy


def _valid_depth_values(depth: np.ndarray, scale: float, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if scale == 0:
        raise ValueError("scale must be non-zero")
    z = np.asarray(depth, dtype=np.float64) / scale
    if z.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    valid = valid_depth_mask(z)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != z.shape:
            raise ValueError("mask must match depth shape")
        valid &= user_mask
    return z, valid


def _as_depth_sequence(depth_images: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth_images)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError("depth_images must have shape (H, W) or (N, H, W)")


def _as_rgbd_color_sequence(color_images: np.ndarray, depth_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(color_images)
    count, height, width = depth_shape
    if arr.ndim == 2:
        if count != 1 or arr.shape != (height, width):
            raise ValueError("single-frame color images must match depth shape")
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        if count == 1 and arr.shape[:2] == (height, width):
            arr = arr[None, ...]
        elif arr.shape == depth_shape:
            arr = arr[..., None]
        else:
            raise ValueError("color_images must align with depth_images")
    elif arr.ndim != 4:
        raise ValueError("color_images must have shape (H, W, C), (N, H, W), or (N, H, W, C)")

    if arr.shape[:3] != depth_shape:
        raise ValueError("color_images must align with depth_images")
    return arr


def _as_optional_mask_sequence(masks: np.ndarray | None, depth_shape: tuple[int, int, int]) -> np.ndarray | None:
    if masks is None:
        return None
    arr = np.asarray(masks, dtype=bool)
    count, height, width = depth_shape
    if arr.ndim == 2:
        if count != 1 or arr.shape != (height, width):
            raise ValueError("single-frame masks must match depth shape")
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape == depth_shape:
        return arr
    raise ValueError("masks must have shape (H, W) or (N, H, W)")


def _as_optional_transform_sequence(transforms: np.ndarray | None, count: int) -> np.ndarray | None:
    if transforms is None:
        return None
    arr = np.asarray(transforms, dtype=np.float64)
    if arr.shape in {(3, 3), (3, 4), (4, 4)}:
        return np.repeat(arr[None, ...], count, axis=0)
    if arr.ndim == 3 and arr.shape[0] == count and arr.shape[1:] in {(3, 3), (3, 4), (4, 4)}:
        return arr
    raise ValueError("transforms must have shape (3, 3), (3, 4), (4, 4), or (N, ..., ...)")


def _rgbd_frame_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
    mask: np.ndarray | None,
    color: np.ndarray | None,
    color_columns: int,
    transform: np.ndarray | None,
) -> np.ndarray:
    z, valid = _valid_depth_values(depth, scale=scale, mask=mask)
    rows, cols = np.nonzero(valid)
    columns = 3 + color_columns
    if rows.size == 0:
        return np.empty((0, columns), dtype=np.float64)

    z_valid = z[rows, cols]
    points = np.column_stack((
        (cols - cx) * z_valid / fx,
        (rows - cy) * z_valid / fy,
        z_valid,
    ))
    if transform is not None:
        points = _transform_depth_points(points, transform)
    if color is None:
        return points
    color_values = np.asarray(color)[rows, cols].reshape(rows.size, color_columns)
    return np.column_stack((points, color_values))


def _transform_depth_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (3, 3):
        return points @ matrix.T
    if matrix.shape == (3, 4):
        linear = matrix[:, :3]
        offset = matrix[:, 3]
    elif matrix.shape == (4, 4):
        linear = matrix[:3, :3]
        offset = matrix[:3, 3]
    else:
        raise ValueError("transform must have shape (3, 3), (3, 4), or (4, 4)")
    return points @ linear.T + offset
