from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import Logger
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback

from main import TimeSeriesModel, TimeSeriesDataModule


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


cli = LightningCLI(TimeSeriesModel, TimeSeriesDataModule, save_config_callback=LoggerSaveConfigCallback)