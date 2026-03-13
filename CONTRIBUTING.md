# Contributing to FRAME-FM

Thank you for your interest in contributing to **FRAME-FM** (Framework for the Rapid development of Environmental Foundation Models)! Contributions of all kinds are welcome — bug reports, feature requests, documentation improvements, and code contributions.

Please take a moment to read through these guidelines before submitting your contribution.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue using the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yaml) template. Include as much detail as possible:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behaviour
- Your environment (OS, Python version, package versions)
- Any relevant logs or screenshots

### Suggesting Features

Feature ideas are welcome! Please open an issue using the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yaml) template and describe:

- The problem your feature would solve
- Your proposed solution
- Any alternatives you've considered

### Contributing Code

1. **Fork the repository** and clone your fork locally.
2. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set up the development environment** (instructions can be found below).
4. **Make your changes**, following the coding standards below.
5. **Write or update tests** to cover your changes.
6. **Run the test suite** to make sure everything passes (instructions can be found below).
7. **Commit your changes** with a clear, descriptive commit message (see [Commit Messages](#commit-messages)).
8. **Push to your fork** and open a Pull Request against the `main` branch.

### Improving Documentation

Documentation improvements are always appreciated. This includes:

- Fixing typos or unclear explanations
- Adding examples or tutorials
- Improving docstrings in the source code

## Development Setup

### Prerequisites

- FRAME-FM fork
- Python >= 3.11, < 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Clone your fork
git clone https://github.com/<your-username>/FRAME-FM.git
cd FRAME-FM

# Install uv
pip install uv

# Create virtual environment and install dependencies
uv venv
uv sync

# Install test dependencies
uv sync --extra test

# (Optional) Install data dependencies
uv sync --extra data
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run a specific test file
uv run pytest tests/datasets/test_base_dataset.py
```

## Coding Standards

### Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints where practical.
- Write clear, self-documenting code with descriptive variable and function names.

### Project Structure

- **Source code** lives in `src/FRAME_FM/`.
- **Tests** live in `tests/` and should mirror the source structure.
- **Configuration** uses [Hydra](https://hydra.cc/) and lives in `configs/`.

### Commit Messages

Use clear, descriptive commit messages. We recommend the following format:

```
<type>: <short summary>

<optional longer description>
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without behaviour changes
- `chore`: Build process or tooling changes

Examples:
```
feat: add ERA5 xarray dataloader
fix: correct tensor shape in masked autoencoder
docs: update environment setup instructions
test: add unit tests for base dataset class
```

## Pull Request Process

1. Ensure your PR targets the `main` branch.
2. Fill out the Pull Request template completely.
3. Link any related issues (e.g., `Closes #42`).
4. Make sure all tests pass and there are no linting errors.
5. Keep PRs focused — one feature or fix per PR.
6. Be responsive to review feedback.

A maintainer will review your PR and may request changes. Once approved, your contribution will be merged.

## Questions?

If you have questions about contributing, feel free to open a [Discussion](https://github.com/British-Oceanographic-Data-Centre/FRAME-FM/discussions) on the repository.

Thank you for helping make FRAME-FM better!
