# TODO

## Package And Publish To Python Registries

- [ ] Confirm the package metadata in `pyproject.toml`:
  - [ ] Package name is correct for the registry: `arraydataengine`.
  - [ ] Version is bumped for the release.
  - [ ] Description, README, license, authors, URLs, classifiers, and `requires-python` are accurate.
  - [ ] Optional dependency groups cover supported installs: `dev`, `image`, `ros`, `dem`, `tiledb`, `visualization`, `notebook`, and `ml`.
- [ ] Add release tooling if it is not already installed:

  ```bash
  python -m pip install --upgrade build twine
  ```

- [ ] Run the pre-release checks from a clean working tree:

  ```bash
  python -m pytest -q
  python -m compileall -q ade tests
  git diff --check
  ```

- [ ] Build the source distribution and wheel:

  ```bash
  python -m build
  ```

- [ ] Validate the built artifacts:

  ```bash
  python -m twine check dist/*
  python -m pip install --force-reinstall dist/*.whl
  python -m pytest -q
  ```

- [ ] Publish to TestPyPI first:

  ```bash
  python -m twine upload --repository testpypi dist/*
  ```

- [ ] Verify the TestPyPI install in a fresh virtual environment:

  ```bash
  python -m venv /tmp/ade-testpypi
  /tmp/ade-testpypi/bin/python -m pip install --upgrade pip
  /tmp/ade-testpypi/bin/python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arraydataengine
  /tmp/ade-testpypi/bin/python -c "import ade; print(ade.__file__)"
  ```

- [ ] Create and push the release commit and tag:

  ```bash
  git add pyproject.toml README.md TODO.md
  git commit -m "Release vX.Y.Z"
  git tag vX.Y.Z
  git push origin main --tags
  ```

- [ ] Publish the same checked artifacts to PyPI:

  ```bash
  python -m twine upload dist/*
  ```

- [ ] Verify the PyPI install in a fresh virtual environment:

  ```bash
  python -m venv /tmp/ade-pypi
  /tmp/ade-pypi/bin/python -m pip install --upgrade pip
  /tmp/ade-pypi/bin/python -m pip install arraydataengine
  /tmp/ade-pypi/bin/python -c "import ade; print(ade.__file__)"
  ```

- [ ] Create a GitHub release from the pushed tag and attach the generated `dist/` artifacts.
- [ ] Record the released version, PyPI URL, TestPyPI URL, and release notes in the project README or GitHub release notes.
