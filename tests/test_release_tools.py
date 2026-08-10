from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

from scripts import check_dist, check_release


VERSION = "1.2.3"
METADATA = b"Metadata-Version: 2.4\nName: arraydataengine\nVersion: 1.2.3\n"


def _write_source_tree(root: Path, changelog_heading: str) -> None:
    (root / "arraydataengine").mkdir()
    (root / "arraydataengine" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "arraydataengine"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{changelog_heading}\n", encoding="utf-8"
    )


def test_release_validation_requires_dated_matching_tag(tmp_path, monkeypatch):
    _write_source_tree(tmp_path, "## 1.2.3 - Unreleased")
    monkeypatch.setattr(check_release, "ROOT", tmp_path)

    assert check_release.validate_release() == []
    assert any(
        "release date" in error
        for error in check_release.validate_release("v1.2.3")
    )

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 1.2.3 - 2026-08-10\n", encoding="utf-8"
    )
    assert check_release.validate_release("v1.2.3") == []
    assert any(
        "does not match" in error
        for error in check_release.validate_release("v1.2.4")
    )

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 1.2.3 - 2026-99-99\n", encoding="utf-8"
    )
    assert any(
        "invalid" in error
        for error in check_release.validate_release("v1.2.3")
    )


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, BytesIO(data))


def _write_valid_artifacts(dist_dir: Path) -> None:
    wheel = dist_dir / f"arraydataengine-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("arraydataengine/__init__.py", "")
        archive.writestr("ArrayDataEngine/__init__.py", "")
        archive.writestr(
            f"arraydataengine-{VERSION}.dist-info/METADATA", METADATA
        )

    root = f"arraydataengine-{VERSION}"
    sdist = dist_dir / f"{root}.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        for relative_name in (
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "arraydataengine/__init__.py",
            "ArrayDataEngine/__init__.py",
            "scripts/check_dist.py",
            "scripts/check_release.py",
        ):
            _add_tar_bytes(archive, f"{root}/{relative_name}", b"content\n")
        _add_tar_bytes(archive, f"{root}/PKG-INFO", METADATA)


def test_distribution_validation_checks_exact_artifacts(tmp_path, monkeypatch):
    _write_source_tree(tmp_path, "## 1.2.3 - Unreleased")
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_valid_artifacts(dist_dir)
    monkeypatch.setattr(check_dist, "ROOT", tmp_path)

    assert check_dist.validate_dist(dist_dir) == []

    with zipfile.ZipFile(next(dist_dir.glob("*.whl")), mode="a") as archive:
        archive.writestr("ade/__init__.py", "")
        archive.writestr("../escape", "")
    assert any(
        "legacy ade" in error for error in check_dist.validate_dist(dist_dir)
    )
    assert any(
        "unsafe archive" in error for error in check_dist.validate_dist(dist_dir)
    )
