"""Click entrypoint."""


from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from FRAME_FM.training.train import main as train_main

import click
from pathlib import Path


console = Console()
CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")


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

@app.group()
def train():
    """Launch a model training run."""
    click.echo("Training command invoked.") # This is a placeholder for the time being.

@train.command(
    "run",
    context_settings=dict(ignore_unknown_options=True),
) # Registers train_run as a subcommand of the train group. Names it "run" so the CLI sees it as frame-fm train run

@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Print the resolved Hydra config to screen before training starts.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train_run(verbose: bool, overrides: tuple[str, ...]):
    """ 
    Start a training run.
    Pass any positional arguments to Hydra to override the config.
    This will not modify the YAML files directly, but can modify the configs.
    \b
    Hydra override syntax:
      key=value   overrides an existing value. Raises an error if the key does not exist.
      +key=value  Append new key. Raises an error if the key already exists.
      ++key=value Override or append.
      ~key=value  Remove a key from the config.
    \b
    Examples:
      framefm train run
      framefm train run model=convAE --- overrides an existing value in config.yaml. will swap demo_autoencoder to convAE
      framefm train run data=land_cover_map --- tells Hydra to override the existing key called data with a value 'land_cover_map'.
      framefm train run +experiment=baseline --- tells Hydra to append a new key called experiment to the config with the value baseline.
      framefm train run --verbose model=convAE
    """

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        if verbose:
            console.print(Panel(OmegaConf.to_yaml(cfg), title="Resolved config"))
        train_main(cfg)

    
@app.command()
@click.option("--list", "list_all", is_flag=True)
def config(list_all):
    """Launch configuration entrypoint."""
    
    if list_all:
        show_config_files()



    click.echo("Config command invoked.") # This is a placeholder for the time being.
