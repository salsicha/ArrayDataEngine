#!/usr/bin/env python3
"""Validate the exact wheel and sdist that may be uploaded to PyPI."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import hashlib
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _metadata_errors(raw_metadata: bytes, version: str, artifact: str) -> list[str]:
    metadata = BytesParser(policy=default).parsebytes(raw_metadata)
    errors: list[str] = []
    if metadata["Name"] != "arraydataengine":
        errors.append(
            f"{artifact}: metadata Name is {metadata['Name']!r}, not "
            "'arraydataengine'"
        )
    if metadata["Version"] != version:
        errors.append(
            f"{artifact}: metadata Version is {metadata['Version']!r}, "
            f"not {version!r}"
        )
    return errors


def _unsafe_members(names: list[str]) -> list[str]:
    unsafe = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(name)
    return unsafe


def _check_wheel(wheel: Path, version: str) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        unsafe = _unsafe_members(names)
        if unsafe:
            errors.append(f"{wheel.name}: unsafe archive members: {unsafe!r}")
        links = [
            member.filename
            for member in members
            if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK
        ]
        if links:
            errors.append(f"{wheel.name}: archive links are not allowed: {links!r}")

        required = {
            "arraydataengine/__init__.py",
            "ArrayDataEngine/__init__.py",
        }
        missing = sorted(required.difference(names))
        if missing:
            errors.append(f"{wheel.name}: missing required files: {missing!r}")
        if any(name.startswith("ade/") for name in names):
            errors.append(f"{wheel.name}: contains the legacy ade import package")

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            errors.append(
                f"{wheel.name}: expected one METADATA file, "
                f"found {len(metadata_names)}"
            )
        else:
            errors.extend(
                _metadata_errors(
                    archive.read(metadata_names[0]), version, wheel.name
                )
            )
    return errors


def _check_sdist(sdist: Path, version: str) -> list[str]:
    errors: list[str] = []
    expected_root = f"arraydataengine-{version}"
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        unsafe = _unsafe_members(names)
        if unsafe:
            errors.append(f"{sdist.name}: unsafe archive members: {unsafe!r}")
        special = [
            member.name
            for member in members
            if not member.isfile() and not member.isdir()
        ]
        if special:
            errors.append(
                f"{sdist.name}: links or special files are not allowed: "
                f"{special!r}"
            )

        roots = {PurePosixPath(name).parts[0] for name in names if name}
        if roots != {expected_root}:
            errors.append(
                f"{sdist.name}: expected archive root {expected_root!r}, "
                f"found {sorted(roots)!r}"
            )

        required = {
            f"{expected_root}/LICENSE",
            f"{expected_root}/README.md",
            f"{expected_root}/pyproject.toml",
            f"{expected_root}/arraydataengine/__init__.py",
            f"{expected_root}/scripts/check_dist.py",
            f"{expected_root}/scripts/check_release.py",
            f"{expected_root}/ArrayDataEngine/__init__.py",
        }
        missing = sorted(required.difference(names))
        if missing:
            errors.append(f"{sdist.name}: missing required files: {missing!r}")
        if any(name.startswith(f"{expected_root}/ade/") for name in names):
            errors.append(f"{sdist.name}: contains the legacy ade import package")

        metadata_name = f"{expected_root}/PKG-INFO"
        try:
            metadata_file = archive.extractfile(metadata_name)
            if metadata_file is None:
                raise KeyError(metadata_name)
        except KeyError:
            errors.append(f"{sdist.name}: missing PKG-INFO")
        else:
            errors.extend(
                _metadata_errors(metadata_file.read(), version, sdist.name)
            )
    return errors


def validate_dist(dist_dir: Path) -> list[str]:
    """Return errors found in a release artifact directory."""
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        version = tomllib.load(pyproject_file)["project"]["version"]

    files = sorted(path for path in dist_dir.iterdir() if path.is_file())
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    errors: list[str] = []

    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        errors.append(f"expected exactly one .tar.gz sdist, found {len(sdists)}")
    expected_files = set(wheels + sdists)
    unexpected = [path.name for path in files if path not in expected_files]
    if unexpected:
        errors.append(f"unexpected files in {dist_dir}: {unexpected!r}")

    if len(wheels) == 1:
        errors.extend(_check_wheel(wheels[0], version))
    if len(sdists) == 1:
        errors.extend(_check_sdist(sdists[0], version))

    for artifact in files:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        print(f"{digest}  {artifact.name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=ROOT / "dist",
        help="artifact directory (default: dist)",
    )
    args = parser.parse_args()
    if not args.dist_dir.is_dir():
        print(
            f"error: artifact directory does not exist: {args.dist_dir}",
            file=sys.stderr,
        )
        return 1

    errors = validate_dist(args.dist_dir)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Release artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
