# PyPI Publishing Guide

## Prerequisites

1. **Create PyPI Account**:
   - Register at [https://pypi.org/account/register/](https://pypi.org/account/register/)
   - Register at [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/) (for testing)

2. **Generate API Tokens**:
   - PyPI: Account settings → API tokens → "Add API token"
   - TestPyPI: Same process on test.pypi.org
   - Scope: "Entire account" (or specific to silicon-alloy once published)

3. **Setup Trusted Publishing (Recommended)**:
   - Go to [PyPI Manage Project](https://pypi.org/manage/project/silicon-alloy/settings/publishing/) -> Publishing.
   - Click "Add a new publisher" -> "GitHub".
   - **Owner**: `hybridindie` (or your username)
   - **Repository**: `alloy`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

   *Alternatively, use legacy API Tokens (not recommended for new setups):*
   - Add `PYPI_API_TOKEN` to GitHub Secrets.

## Manual Publishing

### Test Build Locally

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Check the built package
twine check dist/*

# View contents
tar -tzf dist/silicon-alloy-*.tar.gz
```

### Publish to Test PyPI

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ silicon-alloy
```

### Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Verify
pip install silicon-alloy
```

## Automated Publishing (GitHub Actions)

### Via GitHub Release

1. **Update version** in `pyproject.toml`
2. **Commit and push** changes
3. **Create a release** on GitHub:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Create release from tag on GitHub
5. Workflow automatically publishes to PyPI

### Manual Trigger (Test PyPI)

1. Go to Actions → "Publish to PyPI"
2. Click "Run workflow"
3. Select branch and check "Publish to Test PyPI"
4. Run workflow

## Version Management

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Bump version here
```

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `0.1.0` → Initial release
- `0.1.1` → Bug fixes
- `0.2.0` → New features
- `1.0.0` → Stable API

## Pre-release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Test on Test PyPI first
- [ ] Clean `dist/` directory before building

## Troubleshooting

**"File already exists"**
- You can't overwrite existing versions on PyPI
- Bump the version number and rebuild

**"Invalid authentication"**
- Check your API token is correct
- Ensure token has correct permissions

**"Package name already taken"**
- Choose a different name in `pyproject.toml`
- Or request transfer from current owner

## Post-Publication

After publishing:

1. **Verify installation**:
   ```bash
   pip install silicon-alloy
   metal-diffusion --help
   ```

2. **Update README badge**:
   ```markdown
   ![PyPI](https://img.shields.io/pypi/v/silicon-alloy)
   ```

3. **Announce release** (Twitter, Discord, etc.)
