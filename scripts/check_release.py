#!/usr/bin/env python3
"""Validate source metadata before building or publishing a release."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def validate_release(tag: str | None = None) -> list[str]:
    """Return release metadata errors found in the source tree."""
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    name = project.get("name")
    version = project.get("version")
    errors: list[str] = []

    if name != "arraydataengine":
        errors.append(
            f"project.name must be 'arraydataengine', found {name!r}"
        )
    if not isinstance(version, str) or not version:
        errors.append("project.version must be a non-empty string")
        return errors
    if "+" in version:
        errors.append("public registry releases cannot use a local '+' version")

    expected_tag = f"v{version}"
    if tag is not None and tag != expected_tag:
        errors.append(
            f"release tag {tag!r} does not match project version; "
            f"expected {expected_tag!r}"
        )

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    escaped_version = re.escape(version)
    unreleased_pattern = re.compile(
        rf"^## \[?{escaped_version}\]? - Unreleased$", re.MULTILINE
    )
    dated_pattern = re.compile(
        rf"^## \[?{escaped_version}\]? - (\d{{4}}-\d{{2}}-\d{{2}})$",
        re.MULTILINE,
    )
    is_unreleased = unreleased_pattern.search(changelog) is not None
    dated_matches = dated_pattern.findall(changelog)
    is_dated = len(dated_matches) == 1

    if len(dated_matches) > 1:
        errors.append(f"CHANGELOG.md has duplicate dated {version} headings")
    if is_dated:
        try:
            date.fromisoformat(dated_matches[0])
        except ValueError:
            errors.append(
                f"CHANGELOG.md release date {dated_matches[0]!r} is invalid"
            )
            is_dated = False

    if not is_unreleased and not is_dated:
        errors.append(
            f"CHANGELOG.md needs an Unreleased or dated {version} heading"
        )
    if tag is not None and not is_dated:
        errors.append(
            f"CHANGELOG.md must replace the {version} Unreleased heading "
            "with a YYYY-MM-DD release date before publication"
        )

    if not (ROOT / "arraydataengine" / "__init__.py").is_file():
        errors.append("canonical arraydataengine package is missing")
    if (ROOT / "ade").exists():
        errors.append("legacy ade import package must not be distributed")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help="release tag to match against project.version (for example v0.3.0)",
    )
    args = parser.parse_args()
    errors = validate_release(args.tag)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    mode = f"release tag {args.tag}" if args.tag else "source tree"
    print(f"Release metadata verified for {mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
