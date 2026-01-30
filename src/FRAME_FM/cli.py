"""CLI module for FRAME_FM package."""
import click
from pathlib import Path
#from ./training/train.py import main as train_main

@click.command()
@click.option('--config', 
              prompt='Please enter the config file/folder path', 
              help='The config file or folder path to use for training.',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path)
              )
@click.pass_context
def wrapper(ctx, config):    
    """Simple program that greets NAME."""
    click.echo(f"Training with config: {config}")
    hydra_args = ctx.args
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=f"{config}"):   
        cfg = compose(config_name=config, overrides=hydra_args)
        print(cfg)
        #train_main()

if __name__ == '__main__':
    wrapper()