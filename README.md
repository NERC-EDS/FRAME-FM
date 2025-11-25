# frame_project

## Description
This is a Work in progress project for the development of a framework for training geospatial-temporal models within the oceanographic domain.  

There a number of example files in the root directory of the project. 
These showcase combining the packages which were reviewed during the start of the project. 
The [project board](https://github.com/orgs/British-Oceanographic-Data-Centre/projects/3) showcases the full set of packages looked into.

### AI
There is a collection of markdown files in the notes directory that have come from LLMs (mostly Gemini 3 pro high and claude sonnet 3.5) when tasked with writing code for this project. 

## Installation
This project uses UV as a package manager. To install the dependencies run the following command:

```bash
uv sync
```
however if you don't have uv installed you can install the dependencies from the project's pyproject.toml file:
```bash
pip install .
```
The code has been tested on windows 11 and ubuntu 24.04 (via wsl2).

## Usage
Note: all the mlfow examples require a mlflow server to be running. 
once all virutal environments are activated and the dependencies are installed, 
you can run a local service via:

```bash
uv run mlflow server --backend-store-uri file:./mlruns --default-artifact-root file:./mlruns --host 127.0.0.1 --port 5000
```

The example files in the root directory of the project can be run via either:

```bash
uv run python <fileName>.py
```

or

```bash
python <fileName>.py
```

each files has a different set of packages used. The table below provides a high level overview of which packages are used in each file.

| package           | torchx_mlfow_demo.py | train_eurosat.py | train_gp_spationtemporal.py | train_croissant.py | train_tfds_croissant.py |
|-------------------|----------------------|------------------|-----------------------------|--------------------|-------------------------|
| pytorch linghting |           x          |         x        |              x              |          x         |            x            |
| torchx            |           x          |                  |                             |                    |                         |
| hydra             |           x          |         x        |              x              |                    |            x            |
| mlflow            |                      |         x        |              x              |          x         |            x            |
| torchgeo          |                      |         X        |                             |                    |                         |
| gpytorch          |                      |                  |              X              |                    |                         |
| mlcroissant       |                      |                  |                             |          x         |            x            |

## known issues

### torchx_mlfow_demo.py 
No known issues

### train_eurosat.py
No known issues

### train_gp_spationtemporal.py
No known issues - however the user is warned that the defaults for this example can take a long time to run (5+ hours on an ADM??? with RTX 3090). 

### train_croissant.py
When loading the example dataset their are runtime errors around missing data variavbles. 

### train_tfds_croissant.py
When loading the example dataset their are runtime errors, related to casting of the time variables. 