# Publishing to PyPI

## Current release state

- `pyproject.toml` declares version `0.3.0`. This is the planned first PyPI
  and GitHub release; the `0.1.0` and `0.2.0` changelog entries describe
  repository development milestones, not published releases.
- `python -m build` produces a clean sdist + wheel (no data files leak in);
  both pass `twine check`, and the wheel was smoke-tested in a clean venv with
  only NumPy installed.
- `.github/workflows/publish.yml` builds and publishes via **PyPI Trusted
  Publishing** when a GitHub release is published or the workflow is manually
  dispatched.
- `.github/workflows/tests.yml` runs the test suite on pushes and PRs
  using Python 3.12 and 3.13.

Use one publishing path only. Trusted Publishing (Option A) is recommended:
it avoids a long-lived API token and keeps production publication behind the
`pypi` GitHub environment.

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
   and create one named `pypi` (no secrets needed). Add yourself as a required
   reviewer so each production upload needs a manual approval click.

For the first release, confirm that `pyproject.toml` still declares `0.3.0`,
that CI is green on the release commit, and that the working tree is clean.
Then create and push an annotated tag:

```bash
git status --short
git tag -a v0.3.0 -m "arraydataengine 0.3.0"
git push origin main
git push origin v0.3.0
```

Then on GitHub: **Releases → Draft a new release → choose tag v0.3.0 →
Publish release**. The `publish` workflow builds, checks, and uploads to PyPI.
Approve the `pypi` environment deployment when prompted, then verify at
<https://pypi.org/project/arraydataengine/>.

## Option B — manual upload with twine (fallback)

Do not use this path if the GitHub release workflow will also run for the same
version.

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

Build and check the artifacts first, then upload and install the exact candidate
version in a fresh environment:

```bash
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
python -m venv /tmp/arraydataengine-testpypi
/tmp/arraydataengine-testpypi/bin/python -m pip install --upgrade pip
/tmp/arraydataengine-testpypi/bin/python -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "arraydataengine==0.3.0"
/tmp/arraydataengine-testpypi/bin/python -c \
    "import arraydataengine; print(arraydataengine.__version__)"
```

(TestPyPI needs its own account and token at <https://test.pypi.org>.)

## Future releases

1. Bump `version` in `pyproject.toml` (semantic versioning).
2. Add a section to `CHANGELOG.md`.
3. Run the tests, build both artifacts, run `twine check`, and smoke-test the
   wheel in a clean environment.
4. Commit the release, create an annotated `vX.Y.Z` tag, and explicitly push
   `main` and that tag.
5. Publish a GitHub release and approve the `pypi` environment deployment, or
   use the manual fallback. Do not run both publication paths for one version.

## Notes

- The distribution and canonical import name are both `arraydataengine`.
  `ArrayDataEngine` remains available as a convenience facade. The `ade`
  console command is unchanged.
- The sdist intentionally includes `tests/` but not `example/`, `notebooks/`,
  or any bag/mesh data.
- PyPI uploads are immutable: a version number can never be reused, even
  after deletion. If an upload goes out broken, bump the patch version.
- PyPI and TestPyPI are separate registries with separate accounts and tokens.
