# Publishing to PyPI

Everything mechanical is already prepared:

- `pyproject.toml` has complete metadata (SPDX license, keywords, classifiers,
  URLs) and version `0.1.0`; the name `arraydataengine` is unclaimed on PyPI.
- `python -m build` produces a clean sdist + wheel (no data files leak in);
  both pass `twine check`, and the wheel was smoke-tested in a clean venv with
  only NumPy installed.
- `.github/workflows/publish.yml` builds and publishes via **PyPI Trusted
  Publishing** whenever a GitHub release is published.
- `.github/workflows/tests.yml` runs the test suite on pushes and PRs
  (the exact dependency set was verified locally: 122 passed, 1 skipped).

Two manual paths remain — pick one. Trusted Publishing (Option A) is
recommended: no long-lived API token to store or leak.

## Option A — Trusted Publishing via GitHub Actions (recommended)

One-time setup:

1. Create a PyPI account at <https://pypi.org/account/register/> and enable
   two-factor authentication (required for new projects).
2. On PyPI, go to **Your account → Publishing → Add a new pending publisher**
   and enter exactly:
   - PyPI project name: `arraydataengine`
   - Owner: `salsicha`
   - Repository name: `ArrayDataEngine`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. On GitHub, go to the repo's **Settings → Environments → New environment**
   and create one named `pypi` (no secrets needed; optionally add yourself as
   a required reviewer so publishes need a manual approval click).

Release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then on GitHub: **Releases → Draft a new release → choose tag v0.1.0 →
Publish release**. The `publish` workflow builds, checks, and uploads to PyPI.
Verify at <https://pypi.org/project/arraydataengine/>.

## Option B — manual upload with twine

```bash
python -m pip install --upgrade build twine
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

When prompted, use username `__token__` and an API token created at
<https://pypi.org/manage/account/token/>.

## Optional: dry run against TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ arraydataengine
```

(TestPyPI needs its own account at <https://test.pypi.org>.)

## Future releases

1. Bump `version` in `pyproject.toml` (semantic versioning).
2. Add a section to `CHANGELOG.md`.
3. Commit, tag `vX.Y.Z`, push the tag, and publish a GitHub release
   (Option A) or rebuild and `twine upload` (Option B).

## Notes

- The distribution and canonical import name are both `arraydataengine`.
  `ArrayDataEngine` remains available as a convenience facade. The `ade`
  console command is unchanged.
- The sdist intentionally includes `tests/` but not `example/`, `notebooks/`,
  or any bag/mesh data.
- PyPI uploads are immutable: a version number can never be reused, even
  after deletion. If an upload goes out broken, bump the patch version.
