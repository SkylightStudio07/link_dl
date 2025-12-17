import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from c_trainer import ClassificationTrainer
from f_arg_parser import get_parser
from g_crypto_currency_regression_train_lstm import get_btc_krw_data


def get_model(args):
  class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
      super().__init__()

      hidden_size = 256

      self.lstm = nn.LSTM(
        input_size=n_input,
        hidden_size=hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout if args.lstm_layers >= 2 else 0.0,
        batch_first=True
      )

      fc_layers = []
      for _ in range(args.fc_layers - 1):
        fc_layers.append(nn.Linear(hidden_size, hidden_size))
        fc_layers.append(nn.ReLU())

      fc_layers.append(nn.Linear(hidden_size, n_output))
      self.fcn = nn.Sequential(*fc_layers)

    def forward(self, x):
      x, _ = self.lstm(x)
      x = x[:, -1, :]
      x = self.fcn(x)
      return x

  return MyModel(n_input=6, n_output=2)


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta,
    'weight_decay': args.weight_decay,
    'lstm_layers': args.lstm_layers,
    'fc_layers': args.fc_layers,
    'dropout': args.dropout,
  }

  project_name = "lstm_classification_btc_krw"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="btc_krw experiment with lstm",
    tags=["lstm", "classification", "btc_krw"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  train_data_loader, validation_data_loader, _ = get_btc_krw_data(is_regression=False)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  model = get_model(args)
  model.to(device)

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

  classification_trainer = ClassificationTrainer(
    project_name, model, optimizer, train_data_loader, validation_data_loader, None,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_11_lstm_and_its_application/i_crypto_currency_classification_train_lstm.py -p 100 -r 0.00001
