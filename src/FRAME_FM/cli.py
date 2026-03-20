"""Click entrypoint."""


from collections import defaultdict
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from FRAME_FM.training.train import main as train_main

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from pathlib import Path
import os


console = Console()
DEFAULT_CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")
CONFIG_DIR = os.getenv("CONFIG_DIR", DEFAULT_CONFIG_DIR)

def show_config_files() -> None:
    """Output a structured list of the config folders and their YAML contents."""
    sorted_yamls = defaultdict(list)
    for file in Path(CONFIG_DIR).rglob("*.yaml"):
        sorted_yamls[file.parent.name].append(file.name)

    for folder, files in sorted_yamls.items():
        console.print(Panel(", ".join(files), title=f"Folder: {folder}"))
def display_contents_of_config_file(config_file: str) -> None:
    """Display the contents of a config file."""
    files = list(Path(CONFIG_DIR).rglob(config_file))
    if not files:
        click.secho(f"No matching config found for file: {config_file}.", fg="red")
        return

    for file in files:
        console.print(f"File: {file}")
        console.print(
            Syntax(file.open().read(), "yaml", theme="monokai", line_numbers=True)
        )


def view_hydra_defaults() -> None:
    """Display the Hydra default values from the config."""
    with Path("configs/config.yaml").open() as f:
        contents = yaml.safe_load(f.read())

    if (defaults := contents.get("defaults")) is not None:
        console.print(
            Panel(Pretty(defaults), title="Hydra Defaults", expand=False)
        )
    else:
        click.secho("Unable to find Hydra config file: configs/config.yaml", fg="red")


def edit_config_file(config_file: str, key_value_pairs: str) -> None:
    """Edit the config file, raising errors if the format is incorrect, or the key is missing."""
    with Path(config_file).open() as file:
        file = yaml.safe_load(file.read())

    for pair in key_value_pairs.split(","):
        # Verify that the format is correct.
        if ":" not in pair or len(pair.split(":")) != 2:
            raise click.BadParameter("Expected format -> key:value")

        key, value = pair.split(":")
        if key not in file:
            raise click.BadParameter(f"Key '{key}' not found in config.")

        if isinstance(file[key], float):
            file[key] = float(value)
        elif isinstance(file[key], int):
            file[key] = int(value)
        else:
            file[key] = value

    with open(config_file, mode="w") as edited_file:
        edited_file.write(yaml.dump(file))
        
def train_run_with_options(verbose: bool, overrides: tuple[str, ...]) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name="config", overrides=list(overrides),return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            if verbose:
                console.print(Panel(OmegaConf.to_yaml(cfg), title="Resolved config"))
            train_main(cfg)

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

@click.group()
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
    train_run_with_options(verbose, overrides)
    
    
@click.group()
def config():
    """Configuration entrypoint."""
    pass


@config.command(
    "list", help="This will recursively list all config files in the configs directory."
)
def list_configs():
    """List available config files."""
    show_config_files()


@config.command(
    "display", help="Pass the full path to the config file to display its contents."
)
@click.argument("config_file", type=click.Path(dir_okay=False))
def display(config_file):
    """Display the contents of a config file."""
    display_contents_of_config_file(config_file)


@config.command("view-defaults")
def view_defaults():
    """Show Hydra defaults."""
    view_hydra_defaults()


@config.command(
    "edit",
    help=(
        "Edit values in a config file.\n\n"
        "Pass the full path to the config file followed by key-value pairs "
        "in the format key:new_value.\n\n"
        "Please note that this does remove any comments, and changes the formatting.\n"
        "Examples:\n"
        "\nframefm config edit /path/to/file.yaml batch_size:32\n"
        "\nframefm config edit configs/data/eurosat.yaml num_workers:4,test_split:0.1"
    ),
)
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("key_value_pairs")
def edit(config_file, key_value_pairs):
    """Edit values within a specified config file."""
    edit_config_file(config_file=config_file, key_value_pairs=key_value_pairs)


app.add_command(config)
app.add_command(train)