from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path, PurePath

from .base_source import combine_metadata
from .ros_source import DECODE_ERRORS, DEFAULT_SENSOR_TYPES, RosSource, SkippedMessages


_SQLITE_MAGIC = b"SQLite format 3\x00"
_STORAGE_SUFFIXES = (".db3", ".mcap")
_UNPARSED = object()


def natural_key(path) -> list:
    """Sort rec_2 before rec_10 by comparing embedded numbers numerically."""

    name = os.path.basename(str(path))
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def is_sqlite_file(path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def connect_readonly(path) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True)


def sqlite_topics(connection) -> dict[int, tuple[str, str, str | None]]:
    """topic id -> (name, type, serialization_format) from a rosbag2 SQLite file."""

    columns = {row[1] for row in connection.execute("PRAGMA table_info(topics)")}
    type_column = "type" if "type" in columns else "''"
    format_column = "serialization_format" if "serialization_format" in columns else "NULL"
    topics = {}
    for topic_id, name, msgtype, fmt in connection.execute(
        f"SELECT id, name, {type_column}, {format_column} FROM topics ORDER BY id"
    ):
        topics[int(topic_id)] = (str(name), str(msgtype or ""), fmt if isinstance(fmt, str) else None)
    return topics


def _load_yaml(text: str):
    try:
        import yaml
    except ImportError:
        yaml = None
    try:
        if yaml is not None:
            return yaml.safe_load(text)
        from ruamel.yaml import YAML
    except ImportError:
        return _UNPARSED
    except Exception:  # malformed YAML: fall back to the line scanner
        return _UNPARSED
    try:
        return YAML(typ="safe").load(text)
    except Exception:
        return _UNPARSED


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except ValueError:
            return value[1:-1]
    return value


def _scan_metadata_file_names(text: str) -> list[str]:
    names = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        value = line[2:].strip()
        if value.startswith("path:"):
            value = value[len("path:"):]
        value = _unquote(value)
        if value.lower().endswith(_STORAGE_SUFFIXES):
            names.append(PurePath(value).name)
    return names


def metadata_file_names(metadata_path) -> list[str] | None:
    """Storage file names listed by a rosbag2 metadata.yaml; None when there is none."""

    path = Path(metadata_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    data = _load_yaml(text)
    if data is _UNPARSED:
        names = _scan_metadata_file_names(text)
    else:
        names = []
        info = data.get("rosbag2_bagfile_information") if isinstance(data, dict) else None
        if isinstance(info, dict):
            # Like rosbags, only the base name is used (older bags prefix the
            # bag directory name).
            for value in info.get("relative_file_paths") or []:
                if isinstance(value, str):
                    names.append(PurePath(value).name)
            for entry in info.get("files") or []:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    names.append(PurePath(entry["path"]).name)
    return list(dict.fromkeys(names))


class DB3Source(RosSource):
    """rosbag2 recordings: .db3/.mcap files, split bag directories, and SQLite bags.

    Storage is read as a sequence of units:

    - a directory with ``metadata.yaml`` (or a chunk it lists) is read by rosbags;
    - a standalone SQLite file (.db3, or a SQLite ``.bag``) is read by the
      built-in SQLite reader, which also handles rosbridge JSON payloads;
    - a bare ``.mcap`` without ``metadata.yaml`` is opened by rosbags directly;
    - a directory without ``metadata.yaml`` reads its .db3/.mcap files in
      natural order (rec_2 before rec_10).
    """

    SENSOR_TYPES = dict(DEFAULT_SENSOR_TYPES)


    def __init__(self, data_path: str, max_points: int | None = None):
        """Constructor

        `max_points` is the fixed row count PointCloud2 messages are padded to
        (default 30000); larger clouds are skipped with a warning.
        """
        super().__init__(data_path, max_points=max_points)
        self.input_path = data_path
        self._units = None

        if os.path.isdir(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.dirname(data_path) or "."

    def clear_metadata_cache(self) -> None:
        super().clear_metadata_cache()
        self._units = None

    # -- storage layout ---------------------------------------------------

    def _storage_units(self) -> list[tuple[str, Path]]:
        """(kind, path) pairs; kind is "rosbag2" (directory), "sqlite", or "file"."""

        if self._units is None:
            self._units = self._plan_storage_units()
        return self._units

    def _plan_storage_units(self) -> list[tuple[str, Path]]:
        input_path = Path(self.input_path)
        if input_path.is_dir():
            if (input_path / "metadata.yaml").is_file():
                return [("rosbag2", input_path)]
            files = sorted(
                (
                    path
                    for path in input_path.iterdir()
                    if path.is_file() and path.suffix.lower() in _STORAGE_SUFFIXES
                ),
                key=natural_key,
            )
            return [self._file_unit(path) for path in files]

        if not input_path.is_file():
            return []
        names = metadata_file_names(input_path.parent / "metadata.yaml")
        if names and input_path.name in names:
            # A chunk of a split bag reads the whole containing recording.
            return [("rosbag2", input_path.parent)]
        if is_sqlite_file(input_path):
            # Read exactly the requested file; the surrounding directory may
            # hold a different recording.
            return [("sqlite", input_path)]
        if names is not None and not names:
            # metadata.yaml that lists no files: keep the documented behavior
            # of reading the containing rosbag2 directory.
            return [("rosbag2", input_path.parent)]
        return [("file", input_path)]

    @staticmethod
    def _file_unit(path: Path) -> tuple[str, Path]:
        return ("sqlite", path) if is_sqlite_file(path) else ("file", path)

    def _db3_paths(self) -> list[str]:
        """SQLite storage files backing this source (for diagnostics)."""

        paths = []
        for kind, path in self._storage_units():
            if kind == "sqlite":
                paths.append(str(path))
            elif kind == "rosbag2":
                paths.extend(str(db_path) for db_path in self._sqlite_fallback_paths(path))
        return paths

    @staticmethod
    def _sqlite_fallback_paths(directory: Path) -> list[Path]:
        names = metadata_file_names(directory / "metadata.yaml") or []
        paths = [directory / name for name in names if (directory / name).is_file()]
        if not paths:
            paths = sorted(
                (path for path in directory.iterdir() if path.suffix.lower() == ".db3"),
                key=natural_key,
            )
        return [path for path in paths if is_sqlite_file(path)]

    def _open_reader(self, kind: str, path: Path):
        try:
            return self.reader([path])
        except FileNotFoundError as exc:
            if kind != "file":
                raise
            raise FileNotFoundError(
                f"Cannot open {path} without a rosbag2 metadata.yaml: the installed "
                "rosbags version only reads bag directories. Upgrade rosbags or place "
                "the file in a rosbag2 directory next to its metadata.yaml."
            ) from exc

    # -- messages -----------------------------------------------------------

    def messages(self):
        for kind, path in self._storage_units():
            if kind == "sqlite":
                yield from self._sqlite_messages([path])
                continue
            try:
                reader = self._open_reader(kind, path)
            except ModuleNotFoundError as exc:
                fallback = self._sqlite_fallback_paths(path) if kind == "rosbag2" else []
                if exc.name != "rosbags" or not fallback:
                    raise
                yield from self._sqlite_messages(fallback)
                continue
            yield from self._reader_messages(reader)

    def _sqlite_supported(self, msgtype: str, fmt: str | None) -> bool:
        return self._type_supported(msgtype, fmt, rosbag2=True, has_deserializer=False)

    def _sqlite_messages(self, db_paths):
        skipped = SkippedMessages()
        try:
            for db_path in db_paths:
                with closing(connect_readonly(db_path)) as connection:
                    topics = sqlite_topics(connection)
                    wanted = [
                        topic_id
                        for topic_id, (_, msgtype, fmt) in topics.items()
                        if self._sqlite_supported(msgtype, fmt)
                    ]
                    if not wanted:
                        continue
                    placeholders = ", ".join("?" for _ in wanted)
                    rows = connection.execute(
                        f"""
                        SELECT timestamp, topic_id, data
                        FROM messages
                        WHERE topic_id IN ({placeholders})
                        ORDER BY timestamp ASC, id ASC
                        """,
                        wanted,
                    )
                    for log_time, topic_id, rawdata in rows:
                        topic, msgtype, _ = topics[int(topic_id)]
                        try:
                            message = self._decode_message(
                                topic, msgtype, rawdata, log_time, rosbag2=True
                            )
                        except DECODE_ERRORS as exc:
                            # A malformed payload should not make the whole bag
                            # unreadable; skip the row and keep streaming.
                            skipped.record(topic, msgtype, exc)
                            continue
                        if message is not None:
                            yield message
        finally:
            skipped.summarize()

    # -- metadata -----------------------------------------------------------

    def _metadata(self):
        if self._metadata_cache is None:
            self._metadata_cache = combine_metadata(
                [self._unit_metadata(kind, path) for kind, path in self._storage_units()]
            )
        return self._metadata_cache

    def _unit_metadata(self, kind: str, path: Path) -> dict:
        if kind == "sqlite":
            return self._sqlite_file_metadata(path)
        try:
            reader = self._open_reader(kind, path)
        except ModuleNotFoundError as exc:
            fallback = self._sqlite_fallback_paths(path) if kind == "rosbag2" else []
            if exc.name != "rosbags" or not fallback:
                raise
            return combine_metadata([self._sqlite_file_metadata(db_path) for db_path in fallback])
        return self._reader_metadata(reader)

    def _sqlite_file_metadata(self, db_path) -> dict:
        try:
            with closing(connect_readonly(db_path)) as connection:
                topics = sqlite_topics(connection)
                per_topic = {
                    int(topic_id): int(count)
                    for topic_id, count in connection.execute(
                        "SELECT topic_id, count(*) FROM messages GROUP BY topic_id"
                    )
                }
                start, end = connection.execute(
                    "SELECT min(timestamp), max(timestamp) FROM messages"
                ).fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"{db_path} is not a readable rosbag2 SQLite database: {exc}") from exc

        names = []
        counts = {}
        types = {}
        for topic_id, (topic, msgtype, fmt) in topics.items():
            types.setdefault(topic, msgtype)
            if not self._sqlite_supported(msgtype, fmt):
                continue
            if topic not in counts:
                names.append(topic)
                counts[topic] = 0
            counts[topic] += per_topic.get(topic_id, 0)
        return {
            "topics": names,
            "counts": counts,
            "types": types,
            "start": None if start is None else int(start),
            "end": None if end is None else int(end),
        }


    def data_exists(self) -> bool:
        input_path = Path(self.input_path)
        if input_path.is_dir():
            if (input_path / "metadata.yaml").is_file():
                return True
            return any(
                path.is_file() and path.suffix.lower() in _STORAGE_SUFFIXES
                for path in input_path.iterdir()
            )
        # A file path must exist itself; a typo must not fall back to the
        # bag in the containing directory.
        return input_path.is_file()
