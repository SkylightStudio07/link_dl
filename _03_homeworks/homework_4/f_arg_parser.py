import argparse


def get_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=2048, help="Batch size (int, default: 2_048)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000, help="Number of training epochs (int, default:10_000)"
  )

  parser.add_argument(
    "-r", "--learning_rate", type=float, default=1e-4, help="Learning rate (float, default: 1e-3)"
  )

  parser.add_argument(
    "-w", "--weight_decay", type=float, default=0.0, help="Weight decay (float, default: 0.0)"
  )

  parser.add_argument(
    "-v", "--validation_intervals", type=int, default=30,
    help="Number of training epochs between validations (int, default: 10)"
  )

  parser.add_argument(
    "-p", "--early_stop_patience", type=int, default=30,
    help="Number of early stop patience (int, default: 10)"
  )

  parser.add_argument(
    "-d", "--early_stop_delta", type=float, default=0.000001,
    help="Delta value of early stop (float, default: 0.000001)"
  )

  parser.add_argument(
    "--lstm_layers",
    type=int,
    default=1,
    help="LSTM 레이어 수 (int, default: 1), 값이 1 이상일 때 dropout이 적용."
  )

  parser.add_argument(
    "--fc_layers",
    type=int,
    default=1,
    help="LSTM 이후 Linear(FC) 레이어 수 (int, default: 1)"
  )

  parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="LSTM에 적용할 Dropout 비율 (float, default: 0.0). lstm_layers가 2 이상일 때만 적용됨."
  )
  

  return parser
