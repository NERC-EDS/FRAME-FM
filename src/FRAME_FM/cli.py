"""Click entrypoint."""


from collections import defaultdict
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from FRAME_FM.training.train import main as train_main
from pathlib import Path
import os
from typing import Any

import click
import toml
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from pathlib import Path
import os


console = Console()
DEFAULT_CONFIG_DIR = str(Path(__file__).parents[2] / "configs")
CONFIG_DIR = os.getenv("CONFIG_DIR", DEFAULT_CONFIG_DIR)
torchx_config = os.getenv("TORCHX_CONFIG", ".torchxconfig")

def _type_checker_and_conversion(data: Any, value: Any) -> Any:
    """Utility to check value against its source, and convert if necessary.

    Args:
        data: the source data in the TOML or YAML.
        value: The value to check for in the data.

    Returns:
        The value, with a potential conversion to match the source.
    """
    if isinstance(data, float):
        value = float(value)
    elif isinstance(data, int):
        value = int(value)
    elif isinstance(data, bool):
        value = value.lower() in ("true", "1", "yes")
    return value


def show_config_files(torchx_only: bool) -> None:
    """Output a structured list of configuration folders and their YAML contents.

    Args:
        torchx_only: If True, only locate and show the full path to the torchx config.
    """
    if torchx_only:
        if not (config_location := Path(torchx_config)).is_file():
            click.secho(f"Torchx config not located: {torchx_config}", fg="red")
            return

        click.secho(f"torchx config successfully located at {config_location.resolve()}", fg="green")
        return

    sorted_yamls = defaultdict(list)
    for file in Path(CONFIG_DIR).rglob("*.yaml"):
        sorted_yamls[file.parent.name].append(file.name)

    for folder, files in sorted_yamls.items():
        console.print(Panel(", ".join(files), title=f"Folder: {folder}"))


def display_contents_of_config_file(torchx_only: bool, config_file: str) -> None:
    """Display the contents of a config file.

    Args:
        torchx_only: If True, only display the contents of the torchx config.
        config_file: The file to search for within the config directory.
    """
    if torchx_only:
        torchx_file = Path(torchx_config)
        if not torchx_file.is_file():
            raise click.ClickException(f"Torchx config not found: {torchx_config}")
        with torchx_file.open() as f:
            torchx_contents = f.read()
        console.print(
            Syntax(torchx_contents, "ini", theme="monokai", line_numbers=True))
        return

    files = list(Path(CONFIG_DIR).rglob(config_file))
    if not files:
        click.secho(f"No matching config found for file: {config_file}.", fg="red")
        return

    for file in files:
        console.print(f"File: {file}")
        with file.open() as f:
            contents = f.read()
        console.print(
            Syntax(contents, "yaml", theme="monokai", line_numbers=True)
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
    with Path(config_file).open() as f:
        data = yaml.safe_load(f.read())

    for pair in key_value_pairs.split(","):
        # Verify that the format is correct.
        if ":" not in pair or len(pair.split(":")) != 2:
            raise click.BadParameter("Expected format -> key:value")

        key, value = pair.split(":")
        if key not in data:
            raise click.BadParameter(f"Key '{key}' not found in config.")

        data[key] = _type_checker_and_conversion(data=data[key], value=value)

    with open(config_file, mode="w") as edited_file:
        edited_file.write(yaml.dump(data, sort_keys=False))

        data[key] = _type_checker_and_conversion(data=data[key], value=value)

    with open(config_file, mode="w") as edited_file:
        edited_file.write(yaml.dump(data, sort_keys=False))


def train_run_with_options(verbose: bool, overrides: tuple[str, ...]) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=list(overrides),return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            if verbose:
                console.print(Panel(OmegaConf.to_yaml(cfg), title="Resolved config"))
            train_main(cfg)
def edit_torch_config_file(key_value_pairs: str) -> None:
    """Edit the key-value pairs within the torchxconfig TOML.

    Args:
        key_value_pairs: String representations of keys and their new values to be edited.
    """

    split_items = key_value_pairs.split(",")

    if not Path(torchx_config).is_file():
        raise click.ClickException(f"Torchx config not found: {torchx_config}")

    with open(torchx_config) as file:
        toml_file = toml.load(file)

    for item in split_items:
        try:
            keys, value = item.split(":")
            table, key = keys.split("-")
        except ValueError:
            raise click.ClickException("key-value pairs are incorrectly formatted, refer to the help for examples.")

        if table not in toml_file:
            raise click.ClickException(f"Table '{table}' is not present in the torchx config.")
        elif key not in toml_file[table]:
            raise click.ClickException(f"Key '{key}' is not present in the {table} table in the torchx config.")

        toml_file[table][key] = _type_checker_and_conversion(data=toml_file[table][key], value=value)


    with open(torchx_config, mode="w") as file:
        toml.dump(toml_file, file)



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
@click.option("--torchx", is_flag=True, help="Only show and verify the location for the torchx config.")
def list_configs(torchx):
    """List available config files."""
    show_config_files(torchx_only=torchx)


@config.command(
    "display",
    help=(
        "Display the contents of a config file. "
        "Pass the full path to a config file, or use the --torchx flag to display only the torchx config."
    )
)
@click.option(
    "--torchx",
    is_flag=True,
    help="Display only the contents of the torchx config file."
)
@click.argument(
    "config_file",
    type=click.Path(dir_okay=False),
    required=False
)
def display(torchx, config_file):
    """
    Display the contents of either the torchx config file, or a specific config file provided by the user.
    """

    # Require a config file if not using the torchx flag
    if not torchx and config_file is None:
        raise click.ClickException(message="No config file passed!")

    display_contents_of_config_file(torchx_only=torchx, config_file=config_file)


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


@config.command(
    "edit-torchx",
    help=(
        "Edit values in the TorchX config file.\n\n"
        "Key values should take the form of <table>-<key>:<new_value>\n"
        "Example: scheduler-name:new_local_cwd\n"
        "\nThis edits the 'name' key "
        "in the 'scheduler' table.\n\n"
        "Examples:\n"
        "\nframefm config edit-torchx scheduler-name:new_local_cwd\n"
        "\nframefm config edit-torchx defaults-cpu:4"
    )
)
@click.argument("kv")
def edit_torch(kv):
    """Edit the Torch TOML config file."""
    edit_torch_config_file(key_value_pairs=kv)


app.add_command(config)
app.add_command(train)
