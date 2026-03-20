"""Click entrypoint."""

from collections import defaultdict
from pathlib import Path
import os

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax

console = Console()
config_directory = os.getenv("CONFIG_DIR", "configs")
torchx_config = os.getenv("TORCHX_CONFIG", ".torchxconfig")


def show_config_files(torchx_only: bool) -> None:
    """Output a structured list of the config folders and their YAML contents.
    
    Args:
        torchx_only: If True, only locate and show the full path to the torchx config.
    """
    if torchx_only:
        if not (config_location := Path(torchx_config)).is_file():
            click.secho(f"Torchx config not located: {torchx_config}", fg="red")
        click.secho(f"torchx config successfully located at {config_location.resolve()}", fg="green")
        return
    
    sorted_yamls = defaultdict(list)
    for file in Path(config_directory).rglob("*.yaml"):
        sorted_yamls[file.parent.name].append(file.name)

    for folder, files in sorted_yamls.items():
        console.print(Panel(", ".join(files), title=f"Folder: {folder}"))


def display_contents_of_config_file(config_file: str) -> None:
    """Display the contents of a config file."""
    files = list(Path(config_directory).rglob(config_file))
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


@click.command()
def train():
    """Launch a model training run."""
    click.echo("Training command invoked.")  # This is a placeholder for the time being.


@click.group()
def config():
    """Configuration entrypoint."""
    pass


@config.command(
    "list", help="This will recursively list all config files in the configs directory."
)
@click.option("--torchx", is_flag=True, help="Only show and verify the location for the torchx config.")
def list_configs(torchx):
    """List available config files."""
    show_config_files(torchx_only=torchx)


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