import yaml
import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dataloader.timeseries_datamodule import TimeSeriesDataModule



class LSTMForecast(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.lr = lr

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    # Load training configuration
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    
    random.seed(cfg["seed"])

    # Prepare data module
    dm = TimeSeriesDataModule(
        csv_path=cfg["data"]["csv_path"],
        seq_length=cfg["data"]["seq_length"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"]
    )

    # Initialize model
    model_cfg = cfg["model"]
    model = LSTMForecast(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        lr=model_cfg["lr"]
    )

    # Initialize trainer
    trainer = L.Trainer(**cfg["trainer"])

    # Train the model
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()