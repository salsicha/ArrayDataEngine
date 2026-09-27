from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import islice
from typing import Any

import numpy as np

from .core import DatasetQuery, TopicPipeline, TopicView, dataset_query, topic_view


class IteratorDataset:
    """Plain Python re-iterable dataset wrapper."""

    def __init__(self, factory: Callable[[], Iterable[Mapping[str, Any]]], transform: Callable | None = None):
        self._factory = factory
        self.transform = transform

    def __iter__(self):
        for sample in self._factory():
            yield self.transform(sample) if self.transform is not None else sample


class NumpyDataset:
    """Materialized NumPy-backed sequence dataset."""

    def __init__(self, samples: Iterable[Mapping[str, Any]], copy: bool = True):
        self.samples = tuple(_copy_sample(sample) if copy else dict(sample) for sample in samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.samples[index]

    def __iter__(self):
        yield from self.samples

    def as_arrays(self, pad: bool = True, pad_value=0.0) -> dict[str, Any]:
        return collate_samples(self.samples, pad=pad, pad_value=pad_value)


def to_iterator_dataset(samples, transform: Callable | None = None) -> IteratorDataset:
    """Wrap a sample iterable or factory as a plain Python dataset."""

    if isinstance(samples, IteratorDataset):
        if transform is None:
            return samples
        return IteratorDataset(lambda: iter(samples), transform=transform)
    if callable(samples):
        return IteratorDataset(samples, transform=transform)
    return IteratorDataset(lambda: iter(samples), transform=transform)


def to_numpy_dataset(samples, transform: Callable | None = None, copy: bool = True) -> NumpyDataset:
    """Materialize a sample iterator as a NumPy sequence dataset."""

    iterable = to_iterator_dataset(samples, transform=transform)
    return NumpyDataset(iterable, copy=copy)


def to_torch_dataset(samples, transform: Callable | None = None, iterable: bool = True):
    """Wrap samples as a PyTorch Dataset when `torch` is installed.

    The iterable dataset shards samples across `DataLoader` workers
    (worker `i` of `n` yields samples `i, i + n, ...`), so multi-worker
    loading yields every sample exactly once.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError("to_torch_dataset requires the optional `ml` dependencies, including torch") from exc

    if iterable:
        source = to_iterator_dataset(samples, transform=transform)

        class _TorchIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                samples_iter = iter(source)
                worker = torch.utils.data.get_worker_info()
                if worker is not None and worker.num_workers > 1:
                    samples_iter = islice(samples_iter, worker.id, None, worker.num_workers)
                for sample in samples_iter:
                    yield _to_torch_sample(sample, torch)

        return _TorchIterableDataset()

    dataset = to_numpy_dataset(samples, transform=transform)

    class _TorchDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            return _to_torch_sample(dataset[index], torch)

    return _TorchDataset()


def iter_ml_windows(
    dataset,
    size: int | None = None,
    seconds: float | None = None,
    topics: Iterable[str] | str | None = None,
    chunk_size: int = 1024,
    copy: bool = True,
    transform: Callable | None = None,
) -> IteratorDataset:
    """Return a lazy iterator of topic windows for ML input pipelines."""

    query = _as_dataset_query(dataset)
    if topics is not None:
        query = query.select_topics(topics if isinstance(topics, str) else list(topics))

    def factory():
        for topic, pipeline in query.iter_topics():
            for window in pipeline.window(size=size, seconds=seconds, copy=copy).iter_windows(chunk_size=chunk_size):
                sample = {
                    "topic": topic,
                    "id": None if window.ids is None else window.ids.copy(),
                    "name": None if window.ids is None else window.ids.copy(),
                    "ts": window.timestamps.copy(),
                    "data": window.data.copy() if copy else window.data,
                    "metadata": window.metadata,
                }
                yield sample

    return IteratorDataset(factory, transform=transform)


def deterministic_split_indices(
    count: int,
    fractions: tuple[float, ...] = (0.7, 0.15, 0.15),
    names: tuple[str, ...] = ("train", "val", "test"),
    seed: int | None = None,
    shuffle: bool = False,
    groups: Iterable[Any] | None = None,
) -> dict[str, np.ndarray]:
    """Create deterministic row-index splits, optionally keeping groups together.

    Units (rows, or groups in order of first appearance) are assigned to the
    splits in order. With `shuffle=True` the units are permuted with
    `np.random.default_rng(seed)`; `seed=None` uses seed 0 so the split stays
    deterministic (pass a different seed for a different permutation).
    `seed` is ignored when `shuffle` is false. NaN group values form one group.
    """

    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    fractions_array = _normalized_fractions(fractions, names)

    if groups is None:
        units = list(range(count))
        unit_to_indices = {index: np.array([index], dtype=np.int64) for index in units}
    else:
        group_values = [_hashable_group(value) for value in groups]
        if len(group_values) != count:
            raise ValueError("groups must have one entry per row")
        # Dict insertion order keeps first-appearance order (independent of
        # PYTHONHASHSEED) and groups rows in one pass.
        grouped: dict[Any, list[int]] = {}
        for index, value in enumerate(group_values):
            grouped.setdefault(value, []).append(index)
        units = list(grouped)
        unit_to_indices = {unit: np.asarray(indices, dtype=np.int64) for unit, indices in grouped.items()}

    order = list(units)
    if shuffle:
        rng = np.random.default_rng(0 if seed is None else seed)
        rng.shuffle(order)

    counts = _split_counts(len(order), fractions_array)
    result = {}
    cursor = 0
    for name, split_count in zip(names, counts):
        selected_units = order[cursor:cursor + split_count]
        cursor += split_count
        if not selected_units:
            result[name] = np.empty((0,), dtype=np.int64)
            continue
        indices = np.concatenate([unit_to_indices[unit] for unit in selected_units])
        result[name] = np.sort(indices.astype(np.int64, copy=False))
    return result


def split_topic(
    topic_data,
    by: str = "time",
    fractions: tuple[float, ...] = (0.7, 0.15, 0.15),
    names: tuple[str, ...] = ("train", "val", "test"),
    seed: int | None = None,
    shuffle: bool | None = None,
    geography_columns: tuple[int, int] = (0, 1),
    geography_cell_size: float = 0.01,
) -> dict[str, dict]:
    """Split a topic dict into deterministic train/validation/test-style partitions.

    `by="time"` orders rows by timestamp (stable for ties) before splitting,
    so earlier samples land in the first split even when the topic is not
    time-sorted; `by="index"` uses row order. Each split keeps the original
    row order. See `deterministic_split_indices` for `seed`/`shuffle`.
    """

    view = topic_view(topic_data, copy=False)
    groups = _split_groups(topic_data, view, by, geography_columns, geography_cell_size)
    use_shuffle = False if shuffle is None else bool(shuffle)
    split_indices = deterministic_split_indices(
        len(view),
        fractions=fractions,
        names=names,
        seed=seed,
        shuffle=use_shuffle,
        groups=groups,
    )
    if by.lower().replace("-", "_") == "time":
        time_order = np.argsort(np.asarray(view.timestamps, dtype=np.float64), kind="stable")
        split_indices = {name: np.sort(time_order[indices]) for name, indices in split_indices.items()}
    return {name: _select_view_indices(view, indices).as_dict(copy=False) for name, indices in split_indices.items()}


def augment_image(
    image: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_k: int = 0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    noise_std: float = 0.0,
    seed: int | None = None,
    clip: tuple[float, float] | None = None,
    channel_axis="auto",
) -> np.ndarray:
    """Apply deterministic image-style augmentations to image arrays or sequences.

    `channel_axis="auto"` treats a trailing axis of size 1, 3, or 4 on 3-D+
    input as color channels. A gray `(N, H, W)` sequence whose width is 1, 3,
    or 4 is ambiguous; pass `channel_axis=None` (no color axis: the last two
    axes are spatial) or an integer color axis to be explicit.
    """

    arr = np.asarray(image)
    result = arr.copy()
    vertical_axis, horizontal_axis = _spatial_axes(result, channel_axis)
    if flip_vertical:
        result = np.flip(result, axis=vertical_axis)
    if flip_horizontal:
        result = np.flip(result, axis=horizontal_axis)
    rotate_k = int(rotate_k) % 4
    if rotate_k:
        result = np.rot90(result, k=rotate_k, axes=(vertical_axis, horizontal_axis))
    if contrast != 1.0 or brightness != 0.0 or noise_std:
        values = result.astype(np.float64, copy=False) * float(contrast) + float(brightness)
        if noise_std:
            rng = np.random.default_rng(seed)
            values = values + rng.normal(0.0, float(noise_std), size=values.shape)
        if clip is not None:
            values = np.clip(values, clip[0], clip[1])
        result = _restore_numeric_dtype(values, arr.dtype)
    return np.ascontiguousarray(result)


def augment_dem_patch(
    patch: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_k: int = 0,
    z_scale: float = 1.0,
    z_offset: float = 0.0,
    noise_std: float = 0.0,
    seed: int | None = None,
    channel_axis=None,
) -> np.ndarray:
    """Apply spatial and elevation augmentations to a DEM patch.

    DEM patches have no color axis, so the last two axes are spatial by
    default (a `(N, 3, 3)` patch stack flips each 3x3 patch); see
    `augment_image` for `channel_axis`.
    """

    augmented = augment_image(
        patch,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        rotate_k=rotate_k,
        channel_axis=channel_axis,
    ).astype(np.float64, copy=False)
    if z_scale != 1.0 or z_offset != 0.0 or noise_std:
        augmented = augmented * float(z_scale) + float(z_offset)
        if noise_std:
            rng = np.random.default_rng(seed)
            augmented = augmented + rng.normal(0.0, float(noise_std), size=augmented.shape)
    return _restore_numeric_dtype(augmented, np.asarray(patch).dtype)


def augment_point_cloud(
    points: np.ndarray,
    scale: float = 1.0,
    translation=None,
    rotation=None,
    jitter_std: float = 0.0,
    dropout_ratio: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply XYZ augmentations to point clouds while preserving extra channels."""

    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+)")
    if not 0.0 <= dropout_ratio < 1.0:
        raise ValueError("dropout_ratio must be in [0, 1)")
    rng = np.random.default_rng(seed)
    result = arr.astype(np.float64, copy=True)
    xyz = result[:, :3] * float(scale)
    if rotation is not None:
        matrix = np.asarray(rotation, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        xyz = xyz @ matrix.T
    if translation is not None:
        xyz = xyz + np.asarray(translation, dtype=np.float64)
    if jitter_std:
        xyz = xyz + rng.normal(0.0, float(jitter_std), size=xyz.shape)
    result[:, :3] = xyz
    if dropout_ratio and result.shape[0]:
        keep = rng.random(result.shape[0]) >= float(dropout_ratio)
        if not np.any(keep):
            keep[int(rng.integers(0, result.shape[0]))] = True
        result = result[keep]
    return _restore_numeric_dtype(result, arr.dtype)


def augment_trajectory(
    trajectory: Mapping[str, Any],
    translation=None,
    position_noise_std: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Apply translation and optional position noise to common trajectory mappings."""

    result = _copy_sample(trajectory)
    if "position" not in result:
        return result
    rng = np.random.default_rng(seed)
    position = np.asarray(result["position"], dtype=np.float64).copy()
    if translation is not None:
        position = position + np.asarray(translation, dtype=np.float64)
    if position_noise_std:
        position = position + rng.normal(0.0, float(position_noise_std), size=position.shape)
    result["position"] = position
    if "orientation" in result:
        result["pose"] = np.concatenate((position, np.asarray(result["orientation"], dtype=np.float64)), axis=-1)
    if "trajectory" in result:
        trajectory_array = np.asarray(result["trajectory"], dtype=np.float64).copy()
        trajectory_array[..., :3] = position
        result["trajectory"] = trajectory_array
    if "navsat" in result and "reference" in result:
        from .nav import enu_to_navsat

        reference = result["reference"]
        result["navsat"] = enu_to_navsat(
            position,
            reference["lat"],
            reference["lon"],
            reference.get("alt", 0.0),
        )
    return result


def collate_samples(samples: Iterable[Mapping[str, Any]], pad: bool = True, pad_value=0.0) -> dict[str, Any]:
    """Collate mixed-rate sensor samples, padding variable-size arrays when needed.

    With `pad=True`, every array field with at least one dimension also gets
    `<key>_lengths` (leading-axis lengths) and `<key>_mask` entries, whether
    or not the batch happens to be ragged, so the batch schema is stable.
    Arrays with different numbers of dimensions cannot be padded together and
    raise `ValueError`. `pad_value` is cast to the arrays' dtype when it is
    exactly representable (so int64 nanosecond timestamps stay int64);
    otherwise the dtype is promoted.
    """

    sample_list = [dict(sample) for sample in samples]
    if not sample_list:
        return {}
    keys = sorted({key for sample in sample_list for key in sample})
    batch = {}
    for key in keys:
        values = [sample.get(key) for sample in sample_list]
        _collate_values(batch, key, values, pad=pad, pad_value=pad_value)
    return batch


def _as_dataset_query(dataset) -> DatasetQuery:
    if isinstance(dataset, DatasetQuery):
        return dataset
    if isinstance(dataset, np.ndarray) and dataset.dtype.names is not None and "data" in dataset.dtype.names:
        # A buffered structured topic array (a `DataBuffer.get_buffer()` value).
        return dataset_query({"topic": dataset})
    if isinstance(dataset, Mapping) and "ts" in dataset and "data" in dataset:
        topic = str(dataset.get("topic", "topic"))
        return dataset_query({topic: dataset})
    if isinstance(dataset, Mapping):
        return dataset_query(dataset)
    if isinstance(dataset, (TopicPipeline, TopicView)):
        topic = dataset.metadata.topic if isinstance(dataset, TopicPipeline) else dataset.metadata.topic
        return dataset_query({topic or "topic": dataset})
    raise TypeError("dataset must be a DatasetQuery, topic mapping, topic pipeline, or mapping of topics")


def _copy_sample(sample):
    if isinstance(sample, Mapping):
        return {key: _copy_sample(value) for key, value in sample.items()}
    if isinstance(sample, np.ndarray):
        return sample.copy()
    return sample


def _to_torch_sample(value, torch):
    if isinstance(value, Mapping):
        return {key: _to_torch_sample(item, torch) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.dtype != object:
        return torch.as_tensor(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_torch_sample(item, torch) for item in value)
    return value


def _normalized_fractions(fractions: tuple[float, ...], names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("fractions must be a non-empty one-dimensional sequence")
    if values.size != len(names):
        raise ValueError("fractions and names must have the same length")
    if np.any(values < 0.0) or not np.any(values > 0.0):
        raise ValueError("fractions must be non-negative and include at least one positive value")
    return values / values.sum()


def _split_counts(count: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * count
    counts = np.floor(raw).astype(np.int64)
    remainder = int(count - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


class _NaNGroup:
    """Hashable stand-in so every NaN group value lands in one group."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "nan"


_NAN_GROUP = _NaNGroup()


def _hashable_group(value) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_group(item) for item in value)
    if isinstance(value, (float, complex)) and value != value:
        return _NAN_GROUP
    return value


def _split_groups(topic_data, view: TopicView, by: str, geography_columns, geography_cell_size):
    mode = by.lower().replace("-", "_")
    if mode in {"index", "time"}:
        return None
    if mode == "sequence":
        return view.ids if view.ids is not None else np.arange(len(view), dtype=np.int64)
    if mode in {"source", "source_uri", "file"}:
        if isinstance(topic_data, Mapping):
            source = topic_data.get("source_uri", topic_data.get("source"))
            if source is not None:
                source_array = np.asarray(source, dtype=object)
                if source_array.ndim > 0 and source_array.shape[0] == len(view):
                    return source_array
                return np.full((len(view),), source, dtype=object)
        return np.arange(len(view), dtype=np.int64)
    if mode in {"geography", "geo"}:
        if geography_cell_size <= 0:
            raise ValueError("geography_cell_size must be positive")
        data = np.asarray(view.data, dtype=np.float64)
        if data.ndim < 2 or max(geography_columns) >= data.shape[-1]:
            raise ValueError("geography split requires latitude/longitude columns in topic data")
        coords = np.take(data, geography_columns, axis=-1).reshape((len(view), -1, len(geography_columns)))[:, 0, :]
        buckets = np.floor(coords / float(geography_cell_size)).astype(np.int64)
        groups = np.empty((len(view),), dtype=object)
        groups[:] = [tuple(row) for row in buckets]
        return groups
    raise ValueError("by must be one of time, index, sequence, source, or geography")


def _select_view_indices(view: TopicView, indices: np.ndarray) -> TopicView:
    ids = None if view.ids is None else view.ids[indices]
    return TopicView(ids, view.timestamps[indices], view.data[indices], metadata=view.metadata, copy=True)


def _spatial_axes(arr: np.ndarray, channel_axis="auto") -> tuple[int, int]:
    if arr.ndim < 2:
        raise ValueError("array must have at least two spatial dimensions")
    if isinstance(channel_axis, str):
        if channel_axis != "auto":
            raise ValueError("channel_axis must be 'auto', None, or an integer axis")
        if arr.ndim >= 3 and arr.shape[-1] in {1, 3, 4}:
            return -3, -2
        return -2, -1
    remaining = list(range(arr.ndim))
    if channel_axis is not None:
        axis = int(channel_axis)
        axis = axis + arr.ndim if axis < 0 else axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("channel_axis is out of bounds")
        remaining.remove(axis)
    if len(remaining) < 2:
        raise ValueError("array must have two spatial dimensions besides the channel axis")
    return remaining[-2], remaining[-1]


def _restore_numeric_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(values, info.min, info.max)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    return values


def _collate_values(batch: dict[str, Any], key: str, values: list[Any], pad: bool, pad_value) -> None:
    present = [value for value in values if value is not None]
    if not present:
        batch[key] = np.asarray(values, dtype=object)
        return
    if all(isinstance(value, Mapping) for value in present) and len(present) == len(values):
        batch[key] = collate_samples(values, pad=pad, pad_value=pad_value)
        return
    arrays = [_as_collatable_array(value) for value in values]
    if all(array is not None for array in arrays):
        batch.update(_collate_arrays(key, arrays, pad=pad, pad_value=pad_value))
        return
    batch[key] = np.asarray(values, dtype=object)


def _as_collatable_array(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return None
        return value
    if np.isscalar(value):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except ValueError:
            return None
        return None if array.dtype == object else array
    return None


def _collate_arrays(key: str, arrays: list[np.ndarray], pad: bool, pad_value) -> dict[str, Any]:
    shapes = [array.shape for array in arrays]
    ndims = {array.ndim for array in arrays}
    if not pad:
        if len(set(shapes)) == 1:
            return {key: np.stack(arrays, axis=0)}
        return {key: np.asarray(arrays, dtype=object)}
    if ndims == {0}:
        # Scalars can never be ragged; no lengths/mask needed.
        return {key: np.stack(arrays, axis=0)}
    if len(ndims) != 1:
        raise ValueError(
            f"cannot pad {key!r}: samples have different numbers of dimensions "
            f"({sorted(ndims)}); reshape them consistently or use pad=False"
        )

    max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(len(shapes[0])))
    dtype = _padded_dtype([array.dtype for array in arrays], pad_value)
    padded = np.full((len(arrays), *max_shape), pad_value, dtype=dtype)
    mask = np.zeros((len(arrays), max_shape[0]), dtype=bool)
    lengths = np.zeros((len(arrays),), dtype=np.int64)
    for index, array in enumerate(arrays):
        slices = (index, *[slice(0, size) for size in array.shape])
        padded[slices] = array
        lengths[index] = array.shape[0]
        mask[index, :array.shape[0]] = True
    return {
        key: padded,
        f"{key}_lengths": lengths,
        f"{key}_mask": mask,
    }


def _padded_dtype(dtypes: list[np.dtype], pad_value) -> np.dtype:
    dtype = np.result_type(*dtypes)
    pad = np.asarray(pad_value)
    if dtype.kind in "biufc" and pad.dtype.kind in "biufc":
        try:
            with np.errstate(invalid="ignore", over="ignore"):
                cast = pad.astype(dtype)
            if bool(np.all(cast == pad)):
                return dtype
        except (TypeError, ValueError, OverflowError):
            pass
    return np.result_type(dtype, pad.dtype)
