from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arraydataengine")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"
