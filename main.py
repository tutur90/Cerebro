import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


import pandas as pd
import yaml
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

def custom_ohlc_aggregation(df, window):
    # Shift the DataFrame backward to simulate future window
    shifted = df.shift(-window + 1)

    # Custom aggregations for each column
    open_ = shifted['Open'].rolling(window).apply(lambda x: x.iloc[0], raw=False)
    high_ = shifted['High'].rolling(window).max()
    low_ = shifted['Low'].rolling(window).min()
    close_ = shifted['Close'].rolling(window).apply(lambda x: x.iloc[-1], raw=False)

    return pd.concat([open_, high_, low_, close_], axis=1).dropna().rename(columns={
        'Open': 'TgtOpen', 'High': 'TgtHigh', 'Low': 'TgtLow', 'Close': 'TgtClose'
    })


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distribution, levels, price):

        takes = (levels > price[:, 1].unsqueeze(1)) * (levels < price[:, 2].unsqueeze(1))

        taken = takes * distribution

        pnl = taken * ((levels - price[:, 3].unsqueeze(1))/levels)
        
        pnl = pnl.sum(dim=1)

        pnl = torch.log(pnl.clamp(min=1e-8))

        pnl = pnl.mean()

        return pnl

class TimeSeriesModel(L.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=32, seg_length=60, lr=0.002):
        super().__init__()
        self.save_hyperparameters()  # Saves input args to self.hparams

        self.embedding = nn.Linear(input_dim * seg_length, hidden_dim)
        
        self.relu = nn.ReLU()
        
        
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        
        self.loss_fn = Loss()
        self.seg_length = seg_length
        self.lr = lr
        self.latitude = torch.linspace(-100, 100, steps=output_dim).unsqueeze(0)

    def forward(self, x):
        
        B, T, C = x.shape
        
        num_segments = T // self.seg_length
        
        x_last = x[:, -1, 3].unsqueeze(-1)
        x_centered = x - x_last.unsqueeze(1)
        x_centered = x_centered.view(B, num_segments, self.seg_length * C)
        x_centered = self.embedding(x_centered)
        x_centered = self.relu(x_centered)
        _, (hidden, _) = self.lstm(x_centered)
        return self.fc(hidden[-1]), x_last * self.latitude

    def _shared_step(self, batch, stage):
        inputs, targets = batch
        outputs, levels = self(inputs)
        loss = self.loss_fn(outputs, levels, targets)
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
    def __init__(self, data, src_length=24, tgt_length=1, seg_length=60):
        """
        Args:
            data (pd.DataFrame): Must contain ['Open', 'High', 'Low', 'Close'].
            src_length (int): Number of time units in input (e.g., 24 hours).
            tgt_length (int): Number of time units in target (e.g., 1 hour).
            seg_length (int): Samples per time unit (e.g., 60 = 1/minute).
        """
        self.src_length = src_length * seg_length
        self.tgt_length = tgt_length * seg_length
        self.data = data.reset_index(drop=True)  # Ensure proper indexing

        self.features = torch.tensor(self.data.values, dtype=torch.float32)

        # Precompute targets
        self.tgt_length = tgt_length * seg_length
        self.src_length = src_length * seg_length

        # Compute OHLC targets efficiently
        target_df = custom_ohlc_aggregation(data[['Open', 'High', 'Low', 'Close']], self.tgt_length)


        self.features = torch.tensor(self.data.values[:-self.src_length], dtype=torch.float32)
        self.targets = torch.tensor(target_df.values[self.src_length:], dtype=torch.float32)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        src = self.features[index : index + self.src_length]
        tgt = self.targets[index]
        return src, tgt

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, csv_path, 
                 src_length=24, 
                 tgt_length=1, 
                 seg_length=60,
                 batch_size=32, val_split=0.2, test_split=0.1, num_workers=4):
        super().__init__()
        self.csv_path = csv_path
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.seg_length = seg_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        
        df = pd.read_csv(self.csv_path, index_col='Date')
        df.sort_index(inplace=True)
        
        train_data, self.test_data = train_test_split(df, test_size=self.test_split, shuffle=False)
        self.train_data, self.val_data = train_test_split(train_data, test_size=self.val_split, shuffle=False)

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = StockDataset(self.train_data,self.src_length,self.tgt_length,self.seg_length)
            self.val_dataset = StockDataset(self.val_data,self.src_length,self.tgt_length,self.seg_length)
        if stage == "validate" or stage is None:
            self.val_dataset = StockDataset(self.val_data,self.src_length,self.tgt_length,self.seg_length)
        if stage == "test" or stage is None:
            self.test_dataset = StockDataset(self.test_data,self.src_length,self.tgt_length,self.seg_length)

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
