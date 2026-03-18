"""Click entrypoint."""


from collections import defaultdict
from rich.console import Console
from rich.panel import Panel

import click
from pathlib import Path


console = Console()



def show_config_files() -> None:
    """Output a structured list of the config folders and their YAML contents."""
    sorted_yamls = defaultdict(list)
    for file in Path("configs").rglob("*.yaml"):
        sorted_yamls[file.parent.name].append(file.name)

    for folder, files in sorted_yamls.items():
        console.print(Panel(", ".join(files), title=f"Folder: {folder}"))


@click.group()
def app():
    """
    FRAME-FM is an open-source software framework designed to enable the fast,
    scalable, and accessible development of Foundation Models (FMs) for large-scale
    environmental datasets, including petabyte-scale archives held by the UK’s NERC
    Environmental Data Service (EDS).

    GitHub: https://github.com/NERC-EDS/FRAME-FM

    Two commands are available to run on the command line:

    train:
        Launches a training run

    config:
        Launches a configuration run.
    """
    pass

@app.command()
def train():
    """Launch a model training run."""
    click.echo("Training command invoked.") # This is a placeholder for the time being.



@app.command()
@click.option("--list", "list_all", is_flag=True)
def config(list_all):
    """Launch configuration entrypoint."""
    
    if list_all:
        show_config_files()



    click.echo("Config command invoked.") # This is a placeholder for the time being.
