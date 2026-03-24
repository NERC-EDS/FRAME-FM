"""test cli train command"""

from click.testing import CliRunner
from unittest.mock import patch
from FRAME_FM.cli import app


def test_train_run_invokes_training():
    """To test train run command."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main") as mock_train:
        result = runner.invoke(app, ["train", "run"])

    assert result.exit_code == 0
    mock_train.assert_called_once()


def test_train_run_passes_overrides():
    """Check train run with overrides."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main") as mock_train:
        result = runner.invoke(app, ["train", "run", "model=convAE"])

    assert result.exit_code == 0
    # confirm the cfg passed to train_main has the override applied
    cfg = mock_train.call_args[0][0]
    assert cfg.model._target_ == "FRAME_FM.models.demo_convAE.ConvAutoencoder"


def test_train_run_verbose_prints_config():
    """Check train run command with verbose option."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main"):
        result = runner.invoke(app, ["train", "run", "--verbose"])

    assert result.exit_code == 0
    assert "Resolved config" in result.output


def test_train_run_raises_error_for_missing_experiment():
    """pass a missing experiment and test if error is raised."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main"):
        result = runner.invoke(app, ["train", "run", "+experiment=baseline"])

    assert result.exit_code != 0


def test_train_run_removes_logging():
    """test by removing a default config. for example logging."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main") as mock_train:
        result = runner.invoke(app, ["train", "run", "~logging=demo_mlflow"])

    assert result.exit_code == 0
    mock_train.assert_called_once()
    cfg = mock_train.call_args.args[0]
    assert "logging" not in cfg


def test_train_run_appends_new_key():
    """Test to append a new key to the config."""
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main") as mock_train:
        result = runner.invoke(app, ["train", "run", "++new_key=test_value"])

    assert result.exit_code == 0
    mock_train.assert_called_once()
    cfg = mock_train.call_args.args[0]
    assert cfg.new_key == "test_value"


def test_train_run_overrides_existing_key_with_plusplus():
    runner = CliRunner()

    with patch("FRAME_FM.cli.train_main") as mock_train:
        result = runner.invoke(app, ["train", "run", "++seed=123"])

    assert result.exit_code == 0
    cfg = mock_train.call_args.args[0]
    assert cfg.seed == 123
