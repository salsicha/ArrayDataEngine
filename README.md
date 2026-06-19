<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/CyberPhysics/main/icon.png"
    height="70" align="right" alt="" />
</a>

# Array Data Engine

This repo is a python package for managing array data. It facilitates storing and accessing large amounts of array data, and also provides a generator for many types of data (unlike PyTorch).

## Features

 - NumPy semantics for array data
 - Scales beyond memory limits
 - Generators for more data types
 - and more...

 ## Installation

```bash
python -m pip install -e .
```

Optional feature groups are declared in `pyproject.toml`. For local development with the test suite and common image/TileDB workflows:

```bash
python -m pip install -e ".[dev,image,tiledb]"
```

## Usage

(coming soon)

## Source Benchmarks

Run the source adapter benchmarks with:

```bash
python -m pytest tests/test_source_benchmarks.py -q -s
```

Results below were measured on 2026-06-19 with Python 3.14.5 on arm64. Each result is the best of three runs. Bag, DB3, and DEM use mocked readers/network responses so the benchmarks measure adapter overhead without requiring ROS bag files or Earthdata access.

| Source | Workload | Messages | Elapsed | Throughput | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `ImgSource.messages` | temporary 64x64 PNG files read through OpenCV | 200 | 0.010868s | 18,402 msg/s | 54.3 us/msg |
| `BagSource.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000205s | 973,828 msg/s | 1.0 us/msg |
| `DB3Source.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000191s | 1,046,435 msg/s | 1.0 us/msg |
| `DEMSource.messages` | mocked Earthdata zip response with 32x32 HGT tiles | 4 | 0.000082s | 48,780 msg/s | 20.5 us/msg |

## Documentation and Examples

See notebooks for examples

## Why?

Developing intelligent systems often designing algorithms that depend on correlations between heterogeneous and over long context windows. This project is an attempt to address that with a convenient interface.

## TODO:
Mkdocs
