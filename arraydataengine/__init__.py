from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arraydataengine")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

from .buffer import DataBuffer
from .source import DataSources
from .recording import McapRecorder
from .visualizer import Visualizer
from . import ops
from .ops import *  # noqa: F401,F403

__all__ = [
    "__version__",
    "DataBuffer",
    "DataSources",
    "McapRecorder",
    "Visualizer",
    "ops",
    *ops.__all__,
]

from .parquet_recording import ParquetRecording, mcap_to_parquet, parquet_to_mcap
__all__ += ["ParquetRecording", "mcap_to_parquet", "parquet_to_mcap"]
