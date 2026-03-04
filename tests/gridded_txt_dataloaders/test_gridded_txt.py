import pytest
from hydra.core.config_store import ConfigStore
from typing import Any, List, Optional

def hydra_config_fixture():
    from hydra import initialize, compose
    from omegaconf import OmegaConf, DictConfig
    from dataclasses import dataclass, field, MISSING
    import torchvision

    @dataclass(kw_only=True)
    class TrainTransformsConfig:
        _target_: str = "torchvision.transforms.Compose"
        transforms: list = field(default_factory=lambda: [
            {"_target_": "torchvision.transforms.Resize", "size": (64)},
            {"_target_": "torchvision.transforms.RandomHorizontalFlip", "p": (0.5)},
            {"_target_": "torchvision.transforms.ToTensor"},
            {"_target_": "torchvision.transforms.Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ])
    
    @dataclass(kw_only=True)
    class TestTransformsConfig:
        _target_: str = "torchvision.transforms.Compose"
        transforms: list = field(default_factory=lambda: [
            {"_target_": "torchvision.transforms.Resize", "size": (64)},
            {"_target_": "torchvision.transforms.ToTensor"},
            {"_target_": "torchvision.transforms.Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ])

    @dataclass(kw_only=True)
    class ValTransformsConfig:
        _target_: str = "torchvision.transforms.Compose"
        transforms: list = field(default_factory=lambda: [
            {"_target_": "torchvision.transforms.Resize", "size": (64)},
            {"_target_": "torchvision.transforms.ToTensor"},
            {"_target_": "torchvision.transforms.Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ])


    @dataclass(kw_only=True)
    class DataConfigEurosat:
        _target_: str = "FRAME_FM.dataloaders.demo_eurosat.EuroSATDataModule"

        data_root: str = "../data/eurosat"
        batch_size: int = 32
        num_workers: int = 4
        pin_memory: bool = True
        persistent_workers: bool = True

        # BaseDataModule split logic
        split_strategy: str= "fraction"        # "fraction" | "indices" | "none"
        train_split: float = 0.8
        val_split: float = 0.2
        test_split: float = 0.0

        # Explicit indices not used here (fractions take over)
        train_indices: Any  = None
        val_indices: Any  = None
        test_indices: Any  = None

        train_transforms: Any = field(default_factory=TrainTransformsConfig)
        test_transforms: Any = field(default_factory=TestTransformsConfig)
        val_transforms: Any = field(default_factory=ValTransformsConfig)

    @dataclass(kw_only=True)
    class ExperimentConfigEurosat:
        name: str = "base"
        notes: str = "Default experiment configuration."

    @dataclass(kw_only=True)
    class LoggingConfigEurosat:
        _target_: str = "FRAME_FM.training.logging_utils.create_mlflow_logger"
        experiment_name: str = "frame-fm-demo"
        tracking_uri: str = "../experiments/mlruns}"
        run_name: str = "demo_autoencoder"

        tags: list = field(default_factory=lambda: [
            {"project": "FRAME-FM"},
            {"dataset": "EuroSAT"},
            {"model": "EuroSAT Autoencoder"}
        ])

        
    @dataclass(kw_only=True)
    class ModelConfigEurosat:
        _target_: str = "FRAME_FM.models.demo_autoencoder.EuroSATAutoencoder"

        # Architecture
        in_channels: int = 13          # EuroSAT is Full (13 bands) or RGB (3 bands)
        base_channels: int = 32       # maps to your chs[1] / base filters
        latent_dim: int = 256

        # Optimisation hyperparameters
        lr: float = 1e-3
        weight_decay: float = 1e-5

        # Optional, if your BaseModule uses num_classes or similar
        num_classes: int = 17   

    @dataclass(kw_only=True)
    class TrainerConfigEurosat:
        _target_: str = "pytorch_lightning.Trainer"

        max_epochs: int = 1000
        accelerator: str = "auto"      # "auto", "gpu", "cpu"
        devices: str = "auto"          # or e.g. 1, [0], etc.
        precision: int = 32          # or "16-mixed" for AMP

        log_every_n_steps: int = 5
        enable_checkpointing: bool = True
        enable_progress_bar: bool = True
        enable_model_summary: bool = True

        # Optional extra flags:
        # gradient_clip_val: 0.0
        # deterministic: true
        # run_test: true  # flag for evaluating on test set after training
    
    @dataclass(kw_only=True)
    class HydraConfig:
        run:list = field(default_factory=lambda: [
            {"dir": "../outputs/eurosat_demo"},
        ])

        job: list = field(default_factory=lambda: [
            {"name": "test-EuroSAT"},
            {"chdir": "False"}
        ])

    defaults = [
        {"data": "EuroSatData"},
        {"experiment": "EuroSatExperiment"},
        {"model": "EuroSatModel"},
        {"trainer": "EuroSatTrainer"},
        {"logging": "EuroSatLogging"},
        "_self_"]

    @dataclass(kw_only=True)
    class Config:
        defaults: List[Any] = field(default_factory=lambda: defaults)

        # The following fields will be populated by Hydra based on the defaults list
        # and the registered configs in the ConfigStore.
        data: Any = MISSING
        experiment: Any = MISSING
        model: Any = MISSING
        trainer: Any = MISSING
        logging: Any = MISSING
        seed: int = 42
        hydra: Any = MISSING

    cs = ConfigStore.instance()
    # Registering the Config class with the name "config" allows us to compose it later using that name. 
    # This is a common pattern in Hydra to have a main config class that includes all the defaults and 
    # serves as the entry point for configuration composition.

    cs.store(group="data", name="EuroSatData", node=DataConfigEurosat)
    cs.store(group="experiment", name="EuroSatExperiment", node=ExperimentConfigEurosat)
    cs.store(group="model", name="EuroSatModel", node=ModelConfigEurosat)
    cs.store(group="trainer", name="EuroSatTrainer", node=TrainerConfigEurosat)
    cs.store(group="logging", name="EuroSatLogging", node=LoggingConfigEurosat)
    cs.store(name="config", node=Config)

    initialize()

    cfg = compose(config_name="config")

    return cs, cfg

@pytest.fixture
def eurossat_datamodule():
    from hydra.utils import instantiate

    cs, cfg = hydra_config_fixture()

    datamodule = instantiate(cfg.data)

    yield datamodule


def test_dataset(eurossat_datamodule):
    # GOT TO HERE
    assert eurossat_datamodule == 1 # This is wrong, replace with actual assertions relevant to your dataset and dataloader

def test_eurosat_end_to_end():
    from hydra.utils import instantiate
    import pytorch_lightning as pl

    """
    This test will run the full training loop using the gridded_txt_dataloader 
    and the corresponding model, trainer, and logging configurations defined 
    in the Hydra config.
    """

    cs, cfg = hydra_config_fixture()

    # Ensure reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Instantiate DataModule + Model from config
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    # Configure MLflow logger (if provided)
    logger = None
    if "logging" in cfg:
        logger = instantiate(cfg.logging)

    # Instantiate PL Trainer
    trainer = instantiate(cfg.trainer, logger=logger)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Optional: test after training
    if hasattr(cfg.trainer, "run_test") and cfg.trainer.run_test:
        trainer.test(model, datamodule=datamodule)


    # This test will run after the xarray_dataloader_test fixture has completed
    from gridded_txt_dataloaders import grid_txt_end_to_end
    try:
        grid_txt_end_to_end.main() # Call the main function to run the end-to-end test
    except:
        assert True==False  # Replace with actual assertions relevant to your tests





    

# def test_simple():
#     from dataclasses import dataclass, field

#     import hydra
#     from hydra.core.config_store import ConfigStore
#     from omegaconf import MISSING, OmegaConf
#     from hydra import initialize, compose

#     @dataclass
#     class MySQLConfig:
#         db: list = field(default_factory=lambda: [
#             {"driver": "mysql"},
#             {"pass": "secret"},
#             {"user": "omry"}
#         ])


#     @dataclass
#     class PostGreSQLConfig:
#         db: list = field(default_factory=lambda: [
#             {"driver": "postgresql"},
#             {"pass": "drowssap"},
#             {"timeout": 10},
#             {"user": "postgres_user"}
#         ])

#     defaults = [
#         # Load the config "mysql" from the config group "db"
#         {"db": "mysql"}
#     ]

#     @dataclass
#     class Config:
#         # this is unfortunately verbose due to @dataclass limitations
#         defaults: List[Any] = field(default_factory=lambda: defaults)

#         # Hydra will populate this field based on the defaults list
#         db: Any = MISSING

#     cs = ConfigStore.instance()
#     cs.store(group="db", name="mysql", node=MySQLConfig)
#     cs.store(group="db", name="postgresql", node=PostGreSQLConfig)
#     cs.store(name="config", node=Config)

#     initialize()
#     cfg = compose(config_name="config", overrides=["db=postgresql"])

#     pause = 1


# def test_even_simpler():
#     from omegaconf import OmegaConf, DictConfig
#     import hydra

    # @dataclass
    # class MySQLConfig:
    #     host: str = "localhost"
    #     port: int = 3306

    # # Using the type
    # cs.store(name="config1", node=MySQLConfig)
    # # Using an instance, overriding some default values
    # cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
    # # Using a dictionary, forfeiting runtime type safety
    # cs.store(name="config3", node={"host": "localhost", "port": 3308})

    # initialize()
    # compose(config_name="config3")