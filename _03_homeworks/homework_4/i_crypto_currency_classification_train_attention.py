import torch
from torch import nn, optim
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

from c_trainer import ClassificationTrainer
from f_arg_parser import get_parser
from g_crypto_currency_regression_train_attention import get_btc_krw_data, PositionalEncoding


class SelfAttentionClassificationModel(nn.Module):
  """Self-Attention (Transformer Encoder) 기반 분류 모델"""
  def __init__(self, n_input, n_output, d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1):
    super().__init__()
    
    self.d_model = d_model
    
    # 입력 임베딩: n_input -> d_model
    self.input_embedding = nn.Linear(n_input, d_model)
    
    # Positional Encoding
    self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
    
    # Transformer Encoder Layers
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=n_heads,
      dim_feedforward=ff_dim,
      dropout=dropout,
      batch_first=True,
      activation='gelu'
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    # Layer Normalization
    self.layer_norm = nn.LayerNorm(d_model)
    
    # 출력 레이어 (분류용)
    self.output_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_model, n_output)  # n_output = 클래스 수 (2: 상승/하락)
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
    
    # 마지막 타임스텝의 출력 사용 (LSTM과 동일한 방식)
    x = x[:, -1, :]  # [batch_size, d_model]
    
    # 출력 레이어
    x = self.output_layer(x)  # [batch_size, n_output]
    
    return x


def get_model(args):
  """Self-Attention 분류 모델 생성"""
  return SelfAttentionClassificationModel(
    n_input=6,
    n_output=2,  # 분류: 상승(1) / 하락(0)
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

  project_name = "attention_classification_btc_krw"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="btc_krw experiment with self-attention",
    tags=["attention", "transformer", "classification", "btc_krw"],
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
  
  # 모델 파라미터 수 출력
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")

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

  # python i_crypto_currency_classification_train_attention.py --wandb
  # python i_crypto_currency_classification_train_attention.py --d_model 128 --n_heads 8 --attn_layers 4 --ff_dim 256
