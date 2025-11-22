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

import sys
sys.path.append(BASE_PATH)

from c_trainer import ClassificationTrainer
from a_fashion_mnist_data import get_fashion_mnist_data, get_fashion_mnist_test_data
from e_arg_parser import get_parser


def get_cnn_model():
  """
  Lenet에서 배치 정규화 및 드롭아웃이 들어간 깊은 CNN으로 교체. Val_Acc를 90%대까지 끌어들이기 위한 몸부림..
  1) Convolution 레이어를 3개로 늘리고, 각 Convolution 블록 뒤에 BatchNorm 추가
  2) MaxPool과 Dropout 추가로 오버피팅 억제
  """
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      # Conv + BN + ReLU + MaxPool + Dropout
      self.features = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        # 32 x 28 x 28 → 64 x 28 x 28
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        # 64 x 28 x 28 → 64 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),

        # 64 x 14 x 14 → 128 x 14 x 14
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        # 128 x 14 x 14 → 128 x 7 x 7
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),
      )

      # 분류기
      # Flatten → FC 256 → FC 10
      self.classifier = nn.Sequential(
        nn.Flatten(),                 # 128 * 7 * 7 = 6272
        nn.Linear(128 * 7 * 7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_output),
      )

    def forward(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x

  # 입력: 1 x28 x28 / 출력 클래스: 10
  my_model = MyModel(in_channels=1, n_output=10)

  return my_model

def evaluate_on_test(model, device):
    """
    문제 3 함수 / 학습 완료된 모델로 Fashion-MNIST 테스트 데이터(10,000) 샘플에 대한 손실(loss)과 정확도(accuracy)를 계산하는 함수
    """
    # 1) 테스트 데이터 로더
    f_mnist_test_images, test_loader, test_transforms = get_fashion_mnist_test_data()

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    # 2) 테스트 데이터에 대한 손실과 정확도 계산
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = test_transforms(xb)          # train/val과 동일 정규화
            logits = model(xb)
            loss = criterion(logits, yb)
            # 누적
            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()
            total += yb.size(0)
    # 최종 Loss와 Accuracy 계산
    test_loss = total_loss / total
    test_acc = total_correct / total
    # 3) 결과 출력
    print(f"[Test] loss: {test_loss:.4f}, accuracy: {test_acc*100:.2f}%")

    return test_loss, test_acc

def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta
  }

  project_name = "cnn_mnist"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="fashion-mnist experiment with improved cnn",
    tags=["cnn", "fashion-mnist"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
  model = get_cnn_model()
  model.to(device)

  from torchinfo import summary
  summary(model=model, input_size=(1, 1, 28, 28))

  # 옵티마이저 강화: Adam + weight_decay
  # RMSProp는 쓸 때마다 손실이 자꾸 튀겨서 못써먹을 물건이었다
  optimizer = optim.Adam(
    model.parameters(),
    lr=wandb.config.learning_rate,
    weight_decay=5e-5,  # 오버피팅 좀 줄이기
  )

  classification_trainer = ClassificationTrainer(
      project_name, model, optimizer, train_data_loader, validation_data_loader, f_mnist_transforms,
      run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  save_path = os.path.join(CURRENT_FILE_PATH, "best_val_model.pt")
  torch.save(model.state_dict(), save_path)
  print(f"Saved trained model to: {save_path}")

  # 학습 완료된 모델로 Test Accuracy 계산
  test_loss, test_acc = evaluate_on_test(model, device)

  if args.wandb:
      wandb.run.summary["test_loss"] = test_loss
      wandb.run.summary["test_acc"] = test_acc

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)

  # python _01_code/_11_cnn/a_mnist_train_cnn.py --wandb -b 512 -r 1e-3 -v 1 --epochs 30
  # python _01_code/_11_cnn/a_mnist_train_cnn.py --no-wandb -b 512 -r 1e-3 -v 1 --epochs 30
