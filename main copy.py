import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch.loggers import TensorBoardLogger

from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml

import pandas as pd


# Lightning Module for LSTM Model
class TimeSeriesModel(L.LightningModule):
    def __init__(self,  hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.modules = hparams["modules"]
        
        self.optimizer = hparams["optimizer"]

    def forward(self, x):
        
        x_last = x[:, -1, :]
        x = x - x_last.unsqueeze(1)
        _, (hidden, _) = self.modules.lstm(x)
        
        x = self.modules.proj(hidden[-1]) + x_last
        
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer


# Custom dataset for time-series data
class StockDataset(Dataset):
    def __init__(self, data, seq_length=5):
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # data = scaler.fit_transform(data.values)
        # self.data = data

        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            self.data[index:index+self.seq_length],
            self.data[index+self.seq_length],
        )


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, csv_path, seq_length, batch_size, val_split=0.2, test_split=0.1):
        """
        Args:
            csv_path (str): Path to the CSV file containing time series data.
            seq_length (int): Length of the sequences to be generated.
            batch_size (int): Size of the batches for training and validation.
            val_split (float): Fraction of the dataset to be used for validation.
        """ 
        super().__init__()
        self.csv_path = csv_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        
        print(f"Loading data from {self.csv_path}")
        
        df = pd.read_csv(self.csv_path, index_col='Date')
        df.sort_index(inplace=True)
        
        test_data = df[-int(len(df) * self.test_split):]
        train_data = df[:-int(len(df) * self.test_split)]
        val_data = train_data[-int(len(train_data) * self.val_split):]
        train_data = train_data[:-int(len(train_data) * self.val_split)]
        self.train_dataset = StockDataset(data=train_data, seq_length=self.seq_length)
        self.val_dataset = StockDataset(data=val_data, seq_length=self.seq_length)
        self.test_dataset = StockDataset(data=test_data, seq_length=self.seq_length)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "hparams",
        type=str,
        default="configs/train.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="configs/train.yaml",
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    
    # Load hyperparameters file with command-line overrides
    with open(args.hparams, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)
    
    dm = TimeSeriesDataModule(
        **hparams["data"]
        
    )
    # Initialize model
    model = TimeSeriesModel(
        hparams=hparams
    )

    # Train model
    trainer = L.Trainer(**hparams["trainer"])
    trainer.fit(model, dm)