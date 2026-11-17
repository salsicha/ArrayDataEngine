"""Helpers for 16-bit thermal (LWIR) imagery.

Thermal cameras deliver 14/16-bit counts whose scene content usually spans a
narrow band of the full range (a room at ~22000 +/- 50 counts), with the odd
very hot or cold pixel. Full-range conversions such as
``convert_image_dtype(frame, np.uint8)`` crush such a scene to a couple of grey
levels, so display needs automatic gain control (AGC); radiometric cameras
also encode temperature directly ("TLinear").
"""

from __future__ import annotations

import numpy as np

_TEMPERATURE_UNITS = ("kelvin", "celsius", "fahrenheit")
_AGC_METHODS = ("linear", "plateau")
_MAX_AGC_SAMPLES = 2_000_000


def _as_thermal_stack(frames) -> tuple[np.ndarray, bool]:
    arr = np.asarray(frames)
    if arr.ndim == 2:
        return arr[None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError("thermal frames must have shape (H, W) or (N, H, W)")


def _finite_sample(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    if flat.size > _MAX_AGC_SAMPLES:  # a deterministic stride keeps the statistics cheap
        flat = flat[:: int(np.ceil(flat.size / _MAX_AGC_SAMPLES))]
    flat = flat.astype(np.float64, copy=False)
    return flat[np.isfinite(flat)]


def _linear_mapping(sample: np.ndarray, low: float, high: float):
    if sample.size == 0:
        return lambda x: np.zeros(x.shape)
    lo, hi = np.percentile(sample, [low, high])
    span = hi - lo
    if span <= 0:
        return lambda x: np.where(np.isfinite(x), 0.5, 0.0)
    return lambda x: np.clip((x - lo) / span, 0.0, 1.0)


def _plateau_mapping(sample: np.ndarray, plateau: float, bins: int | None):
    if sample.size == 0:
        return lambda x: np.zeros(x.shape)
    lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        return lambda x: np.where(np.isfinite(x), 0.5, 0.0)
    if bins is None:  # one bin per count for integer data, capped for floats and wide ranges
        integral = np.all(np.equal(np.mod(sample, 1.0), 0.0))
        bins = int(min(65536, hi - lo + 1)) if integral else 4096
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    limit = max(1.0, plateau * sample.size) if plateau <= 1.0 else float(plateau)
    clipped = np.minimum(counts, limit)
    cdf = np.cumsum(clipped)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return lambda x: np.interp(x, centres, cdf)


def thermal_agc(
    frames,
    method: str = "linear",
    low: float = 1.0,
    high: float = 99.0,
    plateau: float = 0.01,
    bins: int | None = None,
    per_frame: bool = False,
    dtype=np.uint8,
) -> np.ndarray:
    """Map raw thermal counts to a display range (automatic gain control).

    Args:
        frames: ``(H, W)`` or ``(N, H, W)`` counts (any integer or float dtype;
            NaN pixels map to 0).
        method: ``"linear"`` stretches linearly between the `low`/`high`
            percentiles, so a few hot or cold pixels cannot flatten the scene;
            ``"plateau"`` is plateau histogram equalization (the AGC thermal
            cameras use): each histogram bin is clipped at `plateau` before
            equalizing, which spreads detail across the output range without
            letting large uniform areas (sky, walls) take it all.
        low, high: percentiles of the linear stretch.
        plateau: per-bin cap for ``"plateau"``, as a fraction of the pixels
            (<= 1) or an absolute count (> 1).
        bins: histogram bins for ``"plateau"`` (default: one per count for
            integer data, at most 65536; 4096 for floats).
        per_frame: derive the mapping from each frame separately. The default
            shares one mapping across the sequence, so playback does not
            flicker as the scene changes.
        dtype: ``np.uint8`` (default), ``np.uint16``, or a float dtype for
            values in ``[0, 1]``.

    Returns:
        Frames of the input shape in `dtype`.
    """

    if method not in _AGC_METHODS:
        raise ValueError(f"method must be one of {_AGC_METHODS}, got {method!r}")
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if plateau <= 0:
        raise ValueError("plateau must be positive")
    stack, single = _as_thermal_stack(frames)

    def mapping(values):
        sample = _finite_sample(values)
        if method == "linear":
            return _linear_mapping(sample, low, high)
        return _plateau_mapping(sample, plateau, bins)

    shared = None if per_frame else mapping(stack)
    unit = np.empty(stack.shape, dtype=np.float64)
    for index, frame in enumerate(stack):
        values = frame.astype(np.float64, copy=False)
        mapped = (mapping(frame) if per_frame else shared)(values)
        unit[index] = np.where(np.isfinite(values), mapped, 0.0)

    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        out = unit.astype(dtype)
    elif dtype.kind == "u":
        out = np.rint(unit * np.iinfo(dtype).max).astype(dtype)
    else:
        raise ValueError("dtype must be an unsigned integer or float dtype")
    return out[0] if single else out


def radiometric_to_temperature(counts, scale: float = 0.01, offset: float = 0.0, unit: str = "celsius") -> np.ndarray:
    """Convert radiometric ("TLinear") counts to temperature.

    Radiometric thermal cameras output counts linear in temperature:
    ``kelvin = counts * scale + offset``. FLIR's radiometric Boson and Lepton
    cameras use ``scale=0.01`` (high-resolution TLinear, centikelvin) or
    ``scale=0.1`` (low resolution) with ``offset=0``; other cameras document
    their own scale and offset.

    Returns float64 temperatures in `unit` (``"celsius"``, ``"kelvin"``, or
    ``"fahrenheit"``).
    """

    unit = str(unit).lower()
    if unit not in _TEMPERATURE_UNITS:
        raise ValueError(f"unit must be one of {_TEMPERATURE_UNITS}, got {unit!r}")
    kelvin = np.asarray(counts, dtype=np.float64) * float(scale) + float(offset)
    if unit == "kelvin":
        return kelvin
    celsius = kelvin - 273.15
    return celsius if unit == "celsius" else celsius * 9.0 / 5.0 + 32.0


def temperature_to_radiometric(
    temperature, scale: float = 0.01, offset: float = 0.0, unit: str = "celsius", dtype=np.uint16
) -> np.ndarray:
    """Inverse of `radiometric_to_temperature`: temperatures to (rounded, clipped) counts."""

    unit = str(unit).lower()
    if unit not in _TEMPERATURE_UNITS:
        raise ValueError(f"unit must be one of {_TEMPERATURE_UNITS}, got {unit!r}")
    if scale == 0:
        raise ValueError("scale must be non-zero")
    values = np.asarray(temperature, dtype=np.float64)
    if unit == "fahrenheit":
        values = (values - 32.0) * 5.0 / 9.0
    kelvin = values if unit == "kelvin" else values + 273.15
    counts = np.rint((kelvin - float(offset)) / float(scale))
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        counts = np.clip(counts, info.min, info.max)
    return counts.astype(dtype)


def colorize_thermal(frames, cmap: str = "inferno", method: str = "linear", **agc_options) -> np.ndarray:
    """Render thermal frames in false colour: AGC, then a colormap.

    Returns ``uint8`` RGB frames shaped ``(..., H, W, 3)``. `cmap` names a
    Matplotlib colormap (``"inferno"``, ``"magma"``, ``"gray"``, ...);
    ``"gray"`` also works without Matplotlib. Remaining keyword arguments go to
    `thermal_agc`.
    """

    unit = thermal_agc(frames, method=method, dtype=np.float64, **agc_options)
    if cmap in ("gray", "grey"):
        grey = np.rint(unit * 255.0).astype(np.uint8)
        return np.repeat(grey[..., None], 3, axis=-1)
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError('colorize_thermal needs matplotlib for colormaps other than "gray" '
                          '(pip install "arraydataengine[visualization]")') from exc
    colors = matplotlib.colormaps[cmap](unit)[..., :3]
    return np.rint(colors * 255.0).astype(np.uint8)
