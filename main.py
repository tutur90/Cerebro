import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
import pandas as pd
import yaml
from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger

class TimeSeriesModel(L.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=4, lr=0.002):
        super().__init__()
        self.save_hyperparameters()  # Saves input args to self.hparams
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_last = x[:, -1, :]
        x_centered = x - x_last.unsqueeze(1)
        _, (hidden, _) = self.lstm(x_centered)
        return self.fc(hidden[-1]) + x_last

    def _shared_step(self, batch, stage):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class StockDataset(Dataset):
    def __init__(self, data, seq_length=5):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            self.data[index:index + self.seq_length],
            self.data[index + self.seq_length],
        )


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, csv_path, seq_length, batch_size, val_split=0.2, test_split=0.1, num_workers=4):
        super().__init__()
        self.csv_path = csv_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path, index_col='Date')
        df.sort_index(inplace=True)

        test_len = int(len(df) * self.test_split)
        val_len = int((len(df) - test_len) * self.val_split)

        test_data = df[-test_len:]
        train_data = df[:-test_len]
        val_data = train_data[-val_len:]
        train_data = train_data[:-val_len]

        self.train_dataset = StockDataset(train_data, self.seq_length)
        self.val_dataset = StockDataset(val_data, self.seq_length)
        self.test_dataset = StockDataset(test_data, self.seq_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = ArgumentParser()
    parser.add_argument("hparams", type=str, default="configs/train.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    hparams = load_config(args.hparams)
    
    L.seed_everything(hparams["seed"])

    # Logger setup
    logger = TensorBoardLogger(**hparams["logger"])
    

    # Instantiate DataModule and Model
    dm = TimeSeriesDataModule(**hparams["data"])
    model = TimeSeriesModel(**hparams["model"])

    # Trainer
    trainer = L.Trainer(
        logger=logger,
        **hparams["trainer"]
    )

    trainer.fit(model, dm)
    
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
