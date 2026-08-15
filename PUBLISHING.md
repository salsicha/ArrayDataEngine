# Publishing to PyPI

## Current release state

- `pyproject.toml` declares version `0.3.1`. Version `0.3.0` was the first
  PyPI release and release tag; the `0.1.0` and `0.2.0` changelog entries are
  repository development milestones, not published releases.
- `.github/workflows/publish.yml` builds and verifies every `v*` tag. A
  manual dispatch is build-only unless TestPyPI is explicitly selected.
- Production publication uses PyPI Trusted Publishing, the protected `pypi`
  GitHub environment, and per-file digital attestations. It cannot be started
  by a manual dispatch.
- The exact wheel later uploaded is smoke-tested on Python 3.12, 3.13, and
  3.14. The regular test workflow runs the full suite on the same versions.
- External actions are pinned to full commit SHAs and updated by Dependabot.
  The repository requires SHA pins and gives `GITHUB_TOKEN` read-only access
  by default.

The release pipeline is deliberately staged:

1. Check the distribution name, version, canonical package, annotated release
   tag, and dated changelog entry.
2. Build one sdist and one wheel, run strict metadata checks, inspect archive
   contents, reject unexpected files, and record SHA-256 hashes.
3. Run the full suite on the exact source and install and exercise that exact
   wheel on every supported Python version.
4. Give only the final publishing job an OIDC token.
5. Pause for required human approval, then upload the already verified files
   with attestations.

## One-time registry account setup

The GitHub environments and repository-side protections are configured. Before
the first upload, complete the registry-side identity setup while signed in to
each registry:

1. Create a PyPI account at <https://pypi.org/account/register/>, enable
   two-factor authentication, then open
   <https://pypi.org/manage/account/publishing/>.
2. Add a pending PyPI Trusted Publisher with exactly:
   - PyPI project name: `arraydataengine`
   - Owner: `salsicha`
   - Repository name: `ArrayDataEngine`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. For the optional rehearsal, create a separate TestPyPI account, enable 2FA,
   open <https://test.pypi.org/manage/account/publishing/>, and add the same
   pending publisher except for environment name `testpypi`.
4. Remove any legacy PyPI or TestPyPI API-token secrets from GitHub and revoke
   those tokens at the registries. Trusted Publishing does not use them.

PyPI and TestPyPI are separate registries. Completing one pending publisher
does not configure the other.

## Optional TestPyPI rehearsal

A TestPyPI version can be uploaded only once. Confirm that `0.3.1` is still
available there, then:

1. Open **GitHub → Actions → release → Run workflow**.
2. Select the intended commit and enable **Publish the verified artifacts to
   TestPyPI**.
3. Review the build, full-suite, and Python 3.12–3.14 wheel smoke-test jobs,
   then approve the `testpypi` environment deployment.
4. Verify the exact candidate in a fresh environment:

```bash
python -m venv /tmp/arraydataengine-testpypi
/tmp/arraydataengine-testpypi/bin/python -m pip install --upgrade pip
/tmp/arraydataengine-testpypi/bin/python -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "arraydataengine==0.3.1"
/tmp/arraydataengine-testpypi/bin/python -c \
    "import arraydataengine; print(arraydataengine.__version__)"
```

A manual dispatch with the TestPyPI option disabled only builds and verifies;
it cannot publish to either registry.

## Production release

Before creating the tag, replace `## 0.3.1 - Unreleased` in
`CHANGELOG.md` with the actual release date. From a clean checkout, run:

```bash
python -m pip install --upgrade build twine
python scripts/check_release.py --tag v0.3.1
python -m pytest -q --strict-config --strict-markers tests/
rm -rf dist/
python -m build
python -m twine check --strict dist/*
python scripts/check_dist.py dist
git diff --check
git status --short
```

Commit the dated changelog and any other intentional release changes. Wait for
the `tests` workflow to pass on that commit, then create and explicitly push
an annotated tag:

```bash
git tag -a v0.3.1 -m "arraydataengine 0.3.1"
git push origin main
git push origin v0.3.1
```

The tag starts the `release` workflow. Its production job cannot run for a
branch or manual dispatch, and the `pypi` environment accepts only `v*`
tags. Inspect the completed build, all three full-suite jobs, and all three
wheel smoke-test jobs before approving the environment deployment.

After publication:

1. Verify <https://pypi.org/project/arraydataengine/> and install the exact
   version in a fresh environment.
2. Create a GitHub release from the existing `v0.3.1` tag and publish it.
   Immutable releases then prevent the tag or release assets from being
   silently replaced.
3. Confirm the PyPI page displays attestations for both the wheel and sdist.

## Manual upload fallback

Use this only if Trusted Publishing is unavailable and the tag-triggered
workflow will not publish the same version. Manual uploads lose the protected
environment and automated attestation path.

```bash
python -m pip install --upgrade build twine
python scripts/check_release.py --tag vX.Y.Z
rm -rf dist/
python -m build
python -m twine check --strict dist/*
python scripts/check_dist.py dist
python -m twine upload dist/*
```

When prompted, use username `__token__` and a project-scoped API token. Revoke
the token after the upload.

## Future releases

1. Bump `version` in `pyproject.toml` using semantic versioning.
2. Add the new changelog section as `Unreleased` while developing.
3. Run a build-only manual workflow whenever the release path changes.
4. Optionally publish the final candidate to TestPyPI.
5. Replace `Unreleased` with the release date, run the local checks, commit,
   wait for CI, and push the matching annotated `vX.Y.Z` tag.
6. Approve the protected PyPI deployment only after every verification job
   succeeds, then create the immutable GitHub release.

## Notes

- The distribution and canonical import name are both `arraydataengine`.
  `ArrayDataEngine` remains available as a convenience facade. The `ade`
  console command is unchanged.
- The sdist intentionally includes `tests/` but not `example/`, `notebooks/`,
  or any bag/mesh data.
- Registry uploads are immutable: a version number can never be reused, even
  after deletion. If an upload is broken, bump the patch version.
