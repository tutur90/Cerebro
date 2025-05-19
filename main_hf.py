import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import yaml
from argparse import ArgumentParser
import numpy as np

class TimeSeriesModel(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, labels=None):
        # inputs shape: (batch_size, seq_length, input_dim)
        x_last = inputs[:, -1, :]
        x_centered = inputs - x_last.unsqueeze(1)
        _, (hidden, _) = self.lstm(x_centered)
        outputs = self.fc(hidden[-1]) + x_last

        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(outputs, labels)

        return {"loss": loss, "logits": outputs} if loss is not None else {"logits": outputs}

class StockDataset(Dataset):
    def __init__(self, data, src_length=60, tgt_length=60):
        self.data = data
        self.src_length = src_length
        self.tgt_length = tgt_length
        # Convert data to numpy for consistency
        self.src = data.iloc[:, :].values
        self.tgt = data.rolling(window=tgt_length).mean().shift(-tgt_length).values
        # Remove NaN rows
        valid_indices = ~np.isnan(self.tgt).any(axis=1)
        self.src = self.src[valid_indices]
        self.tgt = self.tgt[valid_indices]

    def __len__(self):
        return len(self.src) - self.src_length - self.tgt_length + 1

    def __getitem__(self, index):
        src_window = self.src[index:index + self.src_length]
        tgt_window = self.tgt[index + self.src_length]
        return {
            "inputs": torch.tensor(src_window, dtype=torch.float32),
            "labels": torch.tensor(tgt_window, dtype=torch.float32)
        }

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = ArgumentParser()
    parser.add_argument("hparams", type=str, default="configs/train_hf.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    hparams = load_config(args.hparams)

    # Set random seed
    torch.manual_seed(hparams["seed"])
    np.random.seed(hparams["seed"])

    # Load and split data
    df = pd.read_csv(hparams["data"]["csv_path"], index_col='Date')
    df.sort_index(inplace=True)

    train_data, test_data = train_test_split(df, test_size=hparams["data"]["test_split"], shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=hparams["data"]["val_split"], shuffle=False)

    # Create datasets
    train_dataset = StockDataset(train_data, hparams["data"]["src_length"], hparams["data"]["tgt_length"])
    val_dataset = StockDataset(val_data, hparams["data"]["src_length"], hparams["data"]["tgt_length"])
    test_dataset = StockDataset(test_data, hparams["data"]["src_length"], hparams["data"]["tgt_length"])

    # Convert to Hugging Face Dataset
    def convert_to_hf_dataset(dataset):
        data = [dataset[i] for i in range(len(dataset))]
        return HFDataset.from_dict({
            "inputs": [item["inputs"].numpy() for item in data],
            "labels": [item["labels"].numpy() for item in data]
        })

    hf_train_dataset = convert_to_hf_dataset(train_dataset)
    hf_val_dataset = convert_to_hf_dataset(val_dataset)
    hf_test_dataset = convert_to_hf_dataset(test_dataset)

    # Instantiate model
    model = TimeSeriesModel(**hparams["model"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=hparams["trainer"]["output_dir"],
        num_train_epochs=hparams["trainer"]["max_epochs"],
        per_device_train_batch_size=hparams["data"]["batch_size"],
        per_device_eval_batch_size=hparams["data"]["batch_size"],
        learning_rate=hparams["model"]["lr"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=hparams["trainer"]["logging_dir"],
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        mse = ((logits - labels) ** 2).mean()
        return {"eval_mse": mse}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Test the model
    test_results = trainer.evaluate(hf_test_dataset)
    print("Test results:", test_results)

if __name__ == "__main__":
    main()