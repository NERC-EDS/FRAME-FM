# Uv Package Manager

Uv is a fast, modern Python package manager that dramatically improves dependency resolution and project reproducibility. It is designed as a drop-in replacement for common pip and venv workflows, while adding efficient environment management and caching.

---

## Features

- Very fast dependency resolution.
- Works with existing Python package workflows.
- Handles virtual environments and caching for you.
- Available on macOS, Linux, and Windows.

---

# Quick Start

## tl;dr
- Create a Python venv and source it
- `pip install uv`
- run `uv init`
- add packages `uv add pytorch pytorch-lighting pandas geotorch xrray`
- commit to version control `git commit -am "project setup"`
  
## Details 
### Install in a virtual environment

This is the simplest way to try Uv alongside an existing Python installation.

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install uv
```

You can now use Uv inside the active virtual environment:

```
uv add requests
uv run python app.py
```
### uv run
This isn't required to run Python code; it's a UV feature that enables more advanced functionality.
like including .env files in your directory to hold information (i.e debugging levels, API keys, user details)
`uv run --env-file .env.production python app.py`
or 
`uv run --env-file .env.local python app.py`
see more at https://docs.astral.sh/uv/reference/cli/#uv-run

### pyproject.toml
This is the file that contains all the information about your project. 
Key here is the list of dependencies that your project requires; these are added automatically when you use `uv run` 

### uv.lock
uv.lock is a cross-platform lockfile that contains exact information about your project's dependencies. 
Unlike pyproject.toml, which specifies the broad requirements of your project, the lockfile contains the exact resolved versions installed in your project environment. 
This file should be checked into version control to enable consistent, reproducible installations across machines.

uv.lock is a human-readable TOML file managed by uv and should not be edited manually.

See the lockfile documentation for more details.

---

### Install the Uv binary on PATH

This installs the Uv application and makes it available globally from your shell. Adjust paths as needed for your platform and shell.

#### Linux

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ensure your local bin directory is on PATH (for example, in `~/.bashrc`, `~/.zshrc`, or similar):

```
export PATH="$HOME/.local/bin:$PATH"
```

Restart your shell or source your profile file so the change takes effect.

After installation, open a new terminal so the PATH update is picked up.

#### Verify the installation

```
uv --version
```

If this prints a version string, Uv is correctly installed and available on PATH.

---

## Usage examples
UV can and likes to manage its own Python installations. If you have installed UV via this route, then add your Python as follows

```
uv python install # install latest (or what is mentioned in .python-version)
uv python install 3.11 # install named version
uv python install 3.11 3.12 # install multiple named versions 
```


Initialise a new project and install dependencies:

```
uv venv
uv init myproject
cd myproject
uv add numpy pandas
```

Run commands in an isolated environment managed by Uv:

```
uv run python main.py
```

Update dependencies to their latest compatible versions:

```
uv update
```

---

## Further documentation

For detailed documentation, advanced usage, and integrations, see the official docs:

- https://docs.astral.sh/uv
