import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import os
import wandb
import math
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from f_arg_parser import get_parser
from g_rnn_trainer import RegressionTrainer
from p__cryptocurrency_dataset_dataloader import get_cryptocurrency_data, \
  CryptoCurrencyDataset


def get_btc_krw_data(sequence_size=10, validation_size=100, test_size=10, is_regression=True):
  X_train, X_validation, X_test, y_train, y_validation, y_test, y_train_date, y_validation_date, y_test_date \
    = get_cryptocurrency_data(
      sequence_size=sequence_size, validation_size=validation_size, test_size=test_size,
      target_column='Close', y_normalizer=1.0e7, is_regression=is_regression
  )

  # X_train.shape: [3212, 10, 5]
  # X_validation.shape: [100, 10, 5]
  # X_test.shape: [10, 10, 5]
  # y_train.shape: [3212]
  # y_validation.shape: [100]
  # y_test.shape: [10]

  train_crypto_currency_dataset = CryptoCurrencyDataset(X=X_train, y=y_train)
  validation_crypto_currency_dataset = CryptoCurrencyDataset(X=X_validation, y=y_validation)
  test_crypto_currency_dataset = CryptoCurrencyDataset(X=X_test, y=y_test)
# print(X_train.shape, X_validation.shape, X_test.shape, y_train.shape, y_validation.shape, y_test.shape, "!!! - 2")
  train_data_loader = DataLoader(
    dataset=train_crypto_currency_dataset, batch_size=wandb.config.batch_size, shuffle=True
  )
  validation_data_loader = DataLoader(
    dataset=validation_crypto_currency_dataset, batch_size=wandb.config.batch_size, shuffle=True
  )
  test_data_loader = DataLoader(
    dataset=test_crypto_currency_dataset, batch_size=len(test_crypto_currency_dataset), shuffle=True
  )

  return train_data_loader, validation_data_loader, test_data_loader


class PositionalEncoding(nn.Module):
  "Transformer에서 사용되는 Positional Encoding"
  def __init__(self, d_model, max_len=5000, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout) # 드롭아웃 레이어
    
    # Positional encoding 생성
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Positional encoding 계산
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # [1, max_len, d_model]
    
    self.register_buffer('pe', pe) # 위치 인코딩 텐서 등록
  # forward
  # 
  def forward(self, x):
    # x: [batch_size, seq_len, d_model]
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)


class SelfAttentionModel(nn.Module):
  "Transformer Encoder 기반 Time-Series 예측 모델"
  def __init__(self, n_input, n_output, d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1):
    super().__init__()
    
    self.d_model = d_model
    
    # 입력 임베딩: n_input -> d_model
    self.input_embedding = nn.Linear(n_input, d_model)
    
    # Positional Encoding
    self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
    
    # Transformer Encoder Layers
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, # 임베딩 차원 
      nhead=n_heads,# 멀티헤드 어텐션 헤드 수
      dim_feedforward=ff_dim, # FFN hidden dim
      dropout=dropout, # 드롭아웃 비율
      batch_first=True, # 배치 차원이 첫 번째
      activation='gelu' # 활성화 함수
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers) # Transformer Encoder 레이어 수
    
    # Layer Normalization
    self.layer_norm = nn.LayerNorm(d_model)
    
    # 출력 레이어
    self.output_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_model, n_output)
    )
  
  def forward(self, x):
    # x: [batch_size, seq_len, n_input]
    
    # 입력 임베딩
    x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
    
    # Positional Encoding 추가
    x = self.positional_encoding(x)
    
    # Transformer Encoder
    x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
    
    # Layer Normalization
    x = self.layer_norm(x)
    
    # 마지막 타임스텝의 출력 사용
    x = x[:, -1, :]  # [batch_size, d_model]
    
    # 출력 레이어
    x = self.output_layer(x)  # [batch_size, n_output]
    
    return x


def get_model(args):
  "Self-Attention 모델 생성"
  return SelfAttentionModel(
    n_input=6,
    n_output=1,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.attn_layers,
    ff_dim=args.ff_dim,
    dropout=args.attn_dropout
  )


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
    'd_model': args.d_model,
    'n_heads': args.n_heads,
    'attn_layers': args.attn_layers,
    'ff_dim': args.ff_dim,
    'attn_dropout': args.attn_dropout,
  }

  project_name = "attention_regression_btc_krw"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="btc_krw experiment with self-attention",
    tags=["attention", "transformer", "regression", "btc_krw"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  train_data_loader, validation_data_loader, _ = get_btc_krw_data()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  model = get_model(args)
  model.to(device)
  
  # 모델 파라미터 수 출력
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

  regression_trainer = RegressionTrainer(
    project_name, model, optimizer, train_data_loader, validation_data_loader, None,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  regression_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)