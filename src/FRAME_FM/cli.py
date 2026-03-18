"""Click entrypoint."""


import click

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
def config():
    """Launch configuration entrypoint."""
    click.echo("Config command invoked.") # This is a placeholder for the time being.
