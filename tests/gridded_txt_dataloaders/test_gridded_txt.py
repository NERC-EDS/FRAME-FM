import pytest
from hydra.core.config_store import ConfigStore
from typing import Any, Optional



# https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py

@pytest.fixture
def hydra_config_fixture():
    from hydra import initialize, compose, initialize_config_module
    from omegaconf import OmegaConf, DictConfig
    from dataclasses import dataclass, field


    @dataclass
    class DataConfig:
        user: str = "test_user"
        num1: int = 10
        num2: int = 20

    @dataclass
    class ModelConfig:
        type: str = "resnet"
        layers: int = 50

    @dataclass
    class TrainerConfig:
        epochs: int = 100
        batch_size: int = 32 
    
    @dataclass
    class LoggingConfig:
        level: str = "INFO"
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @dataclass
    class Defaults:
        data: Any = field(default_factory=DataConfig)
        model: Any = field(default_factory=ModelConfig)
        trainer: Any = field(default_factory=TrainerConfig)
        logging: Any = field(default_factory=LoggingConfig)
        _self_: Optional[Any] = None  # To allow for self-referencing if needed

    class RunConfig:
        dir: str = "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    @dataclass
    class JobConfig:
        name: str = "test_app"
        chdir: bool = True

    @dataclass
    class HydraConfig:
        run: Any = field(default_factory=RunConfig)
        job: Any = field(default_factory=JobConfig)

    @dataclass
    class Config:
        defaults: Any = field(default_factory=Defaults)
        seed: int = 42
        hydra: Any = field(default_factory=HydraConfig)

    

    @dataclass
    class PostgresSQLConfig:
        driver: str = "postgresql"
        user: str = "jieru"
        password: str = "secret"

    @dataclass
    class MySQLConfig:
        host: str = "localhost"
        port: int = 3306

    cs = ConfigStore.instance()
    # Registering the Config class with the name `conf_test` with the config group `db`
    cs.store(name="conftest", node=Config)

    initialize()

    cfg = compose(config_name="conftest")

    @dataclass
    class MySQLConfig:
        host: str = "localhost"
        port: int = 3306

    cs = ConfigStore.instance()

    # Using the type
    cs.store(name="config1", node=MySQLConfig)
    # Using an instance, overriding some default values
    cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
    # Using a dictionary, forfeiting runtime type safety
    cs.store(name="config3", node={"host": "localhost", "port": 3308})

    initialize()
    compose(config_name="config3")


    yield cfg


def test_with_initialize(hydra_config_fixture) -> None:
    from hydra import initialize, compose
    from omegaconf import OmegaConf, DictConfig

    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        # config is relative to a module
        cfg = compose(config_name="conf_test")

        # my_package.main(cfg)

        assert cfg == {
            "app": {"user": "test_user", "num1": 10, "num2": 20},
            "db": {"host": "localhost", "port": 3306},
        }


    print("Config test passed!")

# def test_even_simpler():
#     from omegaconf import OmegaConf, DictConfig
#     import hydra

#     @hydra.main(version_base=None, config_path="conf")
#     def my_app(cfg: DictConfig) -> None:
#         print(OmegaConf.to_yaml(cfg))

#     def simple_func():
#         print("This is a simple function.")

#     cs = ConfigStore.instance()

#     # Using the type
#     cs.store(name="config1", node=MySQLConfig)
#     # Using an instance, overriding some default values
#     cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
#     # Using a dictionary, forfeiting runtime type safety
#     cs.store(name="config3", node={"host": "localhost", "port": 3308})

#     simple_func()
#     my_app()


def test_dataset():
    from FRAME_FM.dataloaders import gridded_txt_dataloader

    assert True==True

# This is a stupid test. But my brain isn't working right now.
def test_end_to_end(hydra_config_fixture):
    # This test will run after the xarray_dataloader_test fixture has completed
    from gridded_txt_dataloaders import grid_txt_end_to_end
    try:
        grid_txt_end_to_end.main() # Call the main function to run the end-to-end test
    except:
        assert True==False  # Replace with actual assertions relevant to your tests
    