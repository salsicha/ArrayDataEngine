from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path, PurePath

from .base_source import combine_metadata
from .cdr import CUSTOM_CDR_TYPES
from .ros_source import (
    DECODE_ERRORS,
    DEFAULT_SENSOR_TYPES,
    EnteredReader,
    RosSource,
    SkippedMessages,
    serialization_format,
)

_logger = logging.getLogger(__name__)

_SQLITE_MAGIC = b"SQLite format 3\x00"
_STORAGE_SUFFIXES = (".db3", ".mcap")
_UNPARSED = object()
_IDL_SEPARATOR = "=" * 80 + "\n"


def natural_key(path) -> list:
    """Sort rec_2 before rec_10 by comparing embedded numbers numerically."""

    name = os.path.basename(str(path))
    # isdecimal, not isdigit: superscripts such as "²" are digits int() rejects.
    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", name)]


def is_sqlite_file(path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def _probe(connection: sqlite3.Connection) -> sqlite3.Connection:
    # Opening is lazy; the first read surfaces errors such as a WAL database
    # that needs a -shm file in a read-only directory.
    connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
    return connection


def _immutable_may_help(error) -> bool:
    # A WAL database in a read-only directory fails with SQLITE_READONLY
    # ("attempt to write a readonly database") or SQLITE_CANTOPEN (its -shm).
    message = str(error).lower()
    return error is not None and ("readonly" in message or "unable to open" in message)


def connect_readonly(path) -> sqlite3.Connection:
    """Open a SQLite file read-only.

    Tries, in order: a ``mode=ro`` URI; ``immutable=1`` (WAL databases in
    read-only directories, which cannot create their -shm file); and a plain
    connection with ``PRAGMA query_only`` for paths SQLite rejects as URIs
    (Windows UNC and ``\\\\?\\`` paths).
    """

    path = Path(path).absolute()
    attempts = []
    try:
        uri = path.as_uri()
    except ValueError:
        uri = None
    if uri is not None:
        attempts.append((uri + "?mode=ro", True, False))
        attempts.append((uri + "?mode=ro&immutable=1", True, True))
    if path.is_file():
        # A plain connect would create a missing file; only use it for files.
        attempts.append((str(path), False, False))

    first_error = None
    for target, use_uri, immutable in attempts:
        if immutable and not _immutable_may_help(first_error):
            # e.g. "database is locked" while a recorder writes: immutable
            # reads could see a half-written file, so never retry that way.
            continue
        connection = None
        try:
            connection = sqlite3.connect(target, uri=use_uri)
            if not use_uri:
                connection.execute("PRAGMA query_only = 1")
            _probe(connection)
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            first_error = first_error or exc
            continue
        if immutable and Path(str(path) + "-wal").exists():
            _logger.warning(
                "%s is opened as immutable (its directory is read-only); changes still in %s-wal are ignored",
                path,
                path.name,
            )
        return connection
    if first_error is None:
        raise sqlite3.OperationalError(f"unable to open database file {path}")
    raise first_error


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


def sqlite_message_definitions(connection) -> list[tuple[str, str, str]]:
    """(type, encoding, definition) rows a rosbag2 SQLite file embeds (schema >= 4)."""

    try:
        rows = connection.execute(
            "SELECT topic_type, encoding, encoded_message_definition FROM message_definitions ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [
        (str(name), str(encoding), str(definition))
        for name, encoding, definition in rows
        if isinstance(name, str) and isinstance(definition, str) and definition.strip()
    ]


class EmbeddedTypes:
    """Message definitions embedded in one SQLite storage file.

    A type's definition (which includes its dependencies) is parsed into a
    rosbags typestore the first time a message of that type needs it; types
    without a usable embedded definition use the default typestore.
    """

    def __init__(self, definitions):
        self._definitions = {
            name: (encoding, definition)
            for name, encoding, definition in definitions
            if encoding in ("ros2msg", "ros2idl")
        }
        self.names = frozenset(self._definitions)
        self._store = None
        self._resolved: dict[str, object] = {}

    def _parse(self, name: str, encoding: str, definition: str) -> dict:
        from rosbags.typesys import get_types_from_idl, get_types_from_msg

        if encoding == "ros2msg":
            return get_types_from_msg(definition, name)
        types = {}
        for section in definition.split(_IDL_SEPARATOR)[1:]:
            _, idl = section.split("\n", 1)
            types.update(get_types_from_idl(idl))
        return types

    def store_for(self, msgtype: str):
        """Typestore holding the embedded definition of `msgtype`, or None."""

        if msgtype in self._resolved:
            return self._resolved[msgtype]
        store = None
        entry = self._definitions.get(msgtype)
        if entry is not None:
            try:
                if self._store is None:
                    from rosbags.typesys import Stores, get_typestore

                    self._store = get_typestore(Stores.EMPTY)
                if msgtype not in self._store.fielddefs:
                    self._store.register(self._parse(msgtype, *entry))
                if msgtype in self._store.fielddefs:
                    store = self._store
            except Exception as exc:  # unparsable or conflicting: use the default store
                _logger.debug("Ignoring embedded definition of %s: %s", msgtype, exc)
        self._resolved[msgtype] = store
        return store

    def has(self, msgtype: str) -> bool:
        """Whether `msgtype` decodes by a definition embedded in the file."""

        return msgtype in self.names and self.store_for(msgtype) is not None

    def deserialize(self, rawdata, msgtype):
        store = self.store_for(msgtype)
        if store is None:
            from ..sensors.base_sensor import _get_typestore

            store = _get_typestore()
        return store.deserialize_cdr(rawdata, msgtype)


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
            # bag directory name). A lone scalar is one entry, not a sequence.
            relative = info.get("relative_file_paths") or []
            if isinstance(relative, str):
                relative = [relative]
            files = info.get("files") or []
            if isinstance(files, dict):
                files = [files]
            for value in relative if isinstance(relative, list) else []:
                if isinstance(value, str):
                    names.append(PurePath(value).name)
            for entry in files if isinstance(files, list) else []:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    names.append(PurePath(entry["path"]).name)
    return list(dict.fromkeys(names))


def metadata_compression_mode(metadata_path) -> str | None:
    """rosbag2 compression mode ("file"/"message") from metadata.yaml; None when uncompressed."""

    try:
        text = Path(metadata_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    data = _load_yaml(text)
    if data is _UNPARSED:
        match = re.search(r"^\s*compression_mode:\s*['\"]?([A-Za-z]*)", text, re.MULTILINE)
        mode = match.group(1) if match else ""
    else:
        info = data.get("rosbag2_bagfile_information") if isinstance(data, dict) else None
        mode = info.get("compression_mode") if isinstance(info, dict) else ""
    mode = str(mode or "").strip().lower()
    return None if mode in ("", "none") else mode


class DB3Source(RosSource):
    """rosbag2 recordings: .db3/.mcap files, split bag directories, and SQLite bags.

    Storage is read as a sequence of units:

    - a directory with ``metadata.yaml`` (or a chunk it lists) is read by
      rosbags; the serialization format recorded in its SQLite files wins over
      metadata.yaml, and if rosbags cannot open the directory its SQLite files
      are read by the built-in reader instead;
    - a standalone SQLite file (.db3, or a SQLite ``.bag``) is read by the
      built-in SQLite reader, which also handles rosbridge JSON payloads;
    - a bare ``.mcap`` without ``metadata.yaml`` is opened by rosbags directly;
    - a directory without ``metadata.yaml`` reads its .db3/.mcap files in
      natural order (rec_2 before rec_10).
    """

    SENSOR_TYPES = dict(DEFAULT_SENSOR_TYPES)


    def __init__(self, data_path: str, max_points: int | None = None, topics=None):
        """Constructor

        `max_points` (default None: no limit) skips point clouds with more
        finite points, with a warning. `topics` restricts reading to those
        topic names.
        """
        data_path = os.fspath(data_path)
        super().__init__(data_path, max_points=max_points, topics=topics)
        self.input_path = data_path
        # The file or directory that was opened (get_data_path()).
        self.data_path = data_path
        self._units = None
        self._formats = {}
        self._fallback_warned = set()

    def clear_metadata_cache(self) -> None:
        super().clear_metadata_cache()
        self._units = None
        self._formats = {}

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

    @classmethod
    def _error_fallback_paths(cls, directory: Path) -> list[Path]:
        """SQLite files the built-in reader can read in place of rosbags.

        Empty when metadata.yaml lists files that are missing (that error
        must surface instead of reading other files in the directory) and for
        compressed recordings, whose payloads only rosbags can decompress.
        """

        metadata = directory / "metadata.yaml"
        names = metadata_file_names(metadata) or []
        if any(not (directory / name).is_file() for name in names):
            return []
        if metadata_compression_mode(metadata) is not None:
            return []
        return cls._sqlite_fallback_paths(directory)

    @staticmethod
    def _formats_disagree(reader, formats: dict) -> bool:
        """Whether metadata.yaml declares another serialization format than the storage."""

        for connection in getattr(reader, "connections", []):
            stored = formats.get(getattr(connection, "topic", None))
            declared = serialization_format(connection)
            if stored and declared and stored != declared:
                return True
        return False

    def _note_fallback(self, path: Path, level: int, message: str, *args) -> None:
        if str(path) not in self._fallback_warned:
            self._fallback_warned.add(str(path))
            _logger.log(level, message, path, *args)

    def _unit_formats(self, kind: str, path: Path) -> dict:
        """topic -> serialization format stored in a rosbag2 directory's SQLite files.

        The mapeverything recorder writes rosbridge JSON payloads but declares
        ``cdr`` in metadata.yaml; the per-topic value in the database is right.
        """

        if kind != "rosbag2":
            return {}
        key = str(path)
        if key not in self._formats:
            formats = {}
            for db_path in self._sqlite_fallback_paths(path):
                try:
                    with closing(connect_readonly(db_path)) as connection:
                        for name, _, fmt in sqlite_topics(connection).values():
                            if fmt:
                                formats.setdefault(name, fmt)
                except sqlite3.Error as exc:
                    _logger.debug("Cannot read topic formats from %s: %s", db_path, exc)
            self._formats[key] = formats
        return self._formats[key]

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

    def _enter_unit_reader(self, kind: str, path: Path):
        """``(entered_reader, None)`` for a rosbags unit, or ``(None, sqlite_paths)``
        when its SQLite files must be read directly (rosbags missing, or unable
        to open the recording, e.g. an embedded definition it cannot parse)."""

        try:
            context = self._open_reader(kind, path)
            reader = context.__enter__()
        except ModuleNotFoundError as exc:
            fallback = self._sqlite_fallback_paths(path) if kind == "rosbag2" else []
            if exc.name != "rosbags" or not fallback:
                raise
            return None, fallback
        except FileNotFoundError:
            raise
        except Exception as exc:
            fallback = self._error_fallback_paths(path) if kind == "rosbag2" else []
            if not fallback:
                raise
            self._note_fallback(
                path,
                logging.WARNING,
                "rosbags cannot open %s (%s: %s); reading its SQLite storage directly",
                type(exc).__name__,
                exc,
            )
            return None, fallback

        formats = self._unit_formats(kind, path)
        if formats and self._formats_disagree(reader, formats):
            # e.g. the mapeverything recorder: rosbridge JSON in the database
            # but "cdr" in metadata.yaml. Newer rosbags versions then match no
            # storage connection and yield nothing, so read the storage directly.
            fallback = self._error_fallback_paths(path)
            if fallback:
                context.__exit__(None, None, None)
                self._note_fallback(
                    path,
                    logging.INFO,
                    "%s: metadata.yaml and its SQLite storage disagree on serialization formats; "
                    "reading the storage directly",
                )
                return None, fallback
        return EnteredReader(context, reader), None

    # -- messages -----------------------------------------------------------

    def messages(self):
        for kind, path in self._storage_units():
            if kind == "sqlite":
                yield from self._sqlite_messages([path])
                continue
            reader, fallback = self._enter_unit_reader(kind, path)
            if reader is None:
                yield from self._sqlite_messages(fallback)
                continue
            stats = {}
            yield from self._reader_messages(reader, self._unit_formats(kind, path), stats)
            if kind == "rosbag2" and stats.get("expected", 0) > 0 and stats.get("read", 0) == 0:
                # rosbags listed messages but yielded none (it matched no
                # storage connection); nothing was yielded, so read directly.
                fallback = self._error_fallback_paths(path)
                if fallback:
                    self._note_fallback(
                        path,
                        logging.WARNING,
                        "%s: rosbags yielded none of its %d listed messages; reading its SQLite storage directly",
                        stats["expected"],
                    )
                    yield from self._sqlite_messages(fallback)

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
                        for topic_id, (name, msgtype, fmt) in topics.items()
                        if self._topic_wanted(name) and self._sqlite_supported(msgtype, fmt)
                    ]
                    if not wanted:
                        continue
                    embedded = EmbeddedTypes(sqlite_message_definitions(connection))
                    deserializer = embedded.deserialize if embedded.names else None
                    custom = {
                        topic_id: embedded.has(topics[topic_id][1])
                        for topic_id in wanted
                        if topics[topic_id][1] in CUSTOM_CDR_TYPES
                    }
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
                        topic_id = int(topic_id)
                        topic, msgtype, fmt = topics[topic_id]
                        try:
                            message = self._decode_message(
                                topic,
                                msgtype,
                                rawdata,
                                log_time,
                                rosbag2=True,
                                deserializer=deserializer,
                                fmt=fmt,
                                embedded=custom.get(topic_id, False),
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
            self._warn_missing_topics(self._metadata_cache)
        return self._metadata_cache

    def _unit_metadata(self, kind: str, path: Path) -> dict:
        if kind == "sqlite":
            return self._sqlite_file_metadata(path)
        reader, fallback = self._enter_unit_reader(kind, path)
        if reader is None:
            return combine_metadata([self._sqlite_file_metadata(db_path) for db_path in fallback])
        return self._reader_metadata(reader, self._unit_formats(kind, path))

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
            if not self._topic_wanted(topic) or not self._sqlite_supported(msgtype, fmt):
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
