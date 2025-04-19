import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, sp
import lightning as L

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
        df = pd.read_csv(self.csv_path, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        data = torch.tensor(df.values, dtype=torch.float32)
        


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
