import os
import sys

from datetime import datetime
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
import argparse
import numpy as np

# 활성화 함수 관련 패러미터 처리

def get_activation(name: str):
    name = (name or "sigmoid").lower()
    if name == "sigmoid":
        return nn.Sigmoid()    
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name in ("leakyrelu", "leaky_relu", "lrelu"):
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")

def get_titanic_dataloaders(batch_size: int):
    """
    titanic_dataset.py에서 제공하는 get_preprocessed_dataset()을 이용.
    - train/valid/test 세 개 데이터셋을 반환받고
    - 그중 train하고 valid만 데이터로더로 묶어서 반환.
    """
    import importlib.util
    TITANIC_DS_PATHS = [
        os.path.join(os.getcwd(), "titanic_dataset.py")
        ]
    ds_path = None
    for p in TITANIC_DS_PATHS:
        if os.path.exists(p):
            ds_path = p
            break
    
    # 오류 발생으로 인한 동적 및 강제 import
    spec = importlib.util.spec_from_file_location("titanic_dataset", ds_path)
    titanic_dataset = importlib.util.module_from_spec(spec)
    sys.modules["titanic_dataset"] = titanic_dataset
    spec.loader.exec_module(titanic_dataset)

    # 전처리된 데이터셋 가져오기
    train_ds, valid_ds, test_ds = titanic_dataset.get_preprocessed_dataset()

    # 파이토치 데이터로더
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=len(valid_ds), shuffle=False)

    # 검증 데이터 배치 하나에서 input_dim(그러니까 특징 개수=10)을 추론
    ex = next(iter(valid_loader))
    input_dim = ex["input"].shape[1]

    return train_loader, valid_loader, input_dim

# ---------- Model ----------

# 원본 캘리포니아 하우스는 회귀라 분류로 수정.
# 또한 출력 차원도 1에서 2(사망/생존)로 수정.
# 처리 용이를 위해 입출력도 딕셔너리로 받아오도록....

class MyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str = "relu", dropout: float = 0.1):
        super().__init__()
        act = get_activation(activation) # ReLU/Sigmoid/ELU/LeakyReLU 중 인자로 받아오는...
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes), # 출력 2
        )

    def forward(self, x):
        return self.net(x)

# 학습 용이화를 위한 파라미터 구조화

@dataclass
class TrainConfig:
    input_dim: int
    num_classes: int = 2
    hidden_dim: int = 64
    activation: str = "relu"
    dropout: float = 0.1
    batch_size: int = 32
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    project: str = "hw2-titanic"
    entity: str | None = None
    run_name: str | None = None

# 학습 루프
# 한 에포크 단위로 학습 -> 검증 -> wandb 로깅

def training_loop(model, optimizer, train_loader, valid_loader, max_epochs=30, device="cpu"):
    # 손실 함수: 분류용으로 변경
    criterion = nn.CrossEntropyLoss() # 분류니까 MSELoss -> CrossEntropyLoss
    model.to(device)
    # 가장 좋은 검증 손실 기록용
    best_state = None
    # 검증 손실 초기화
    best_valid_loss = float("inf")
    global_step = 0
    # 에포크 단위 학습
    for epoch in range(1, max_epochs + 1):
        model.train() # 학습 모드
        train_losses = [] # 에포크 내 배치 손실 기록용
        for batch in train_loader: # 배치 단위 학습
            x = batch["input"].to(device) # 입력
            y = batch["target"].to(device) # 타겟
            # 순전파
            logits = model(x)
            # 손실 계산
            loss = criterion(logits, y)
            # 역전파 및 최적화
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # 배치 손실 기록
            train_losses.append(loss.item())
            global_step += 1

        # 검증. 전체 검증 배치를 한 번에 평가하도록
        # 검증은 기울기 계산 안 함
        model.eval()
        # 검증 손실과 정확도 계산
        with torch.no_grad():
            # 검증 배치 하나 가져오기
            vbatch = next(iter(valid_loader))
            vx = vbatch["input"].to(device)
            vy = vbatch["target"].to(device)
            vlogits = model(vx)
            vloss = criterion(vlogits, vy).item() # 검증 손실
            preds = vlogits.argmax(dim = 1) # 예측 클래스
            vacc = (preds == vy).float().mean().item() # 정확도 계산
        # W&B 로깅
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        wandb.log({
            "epoch": epoch,
            "Training loss": mean_train_loss,
            "Validation loss": vloss,
            "Validation acc": vacc,
        }, step=global_step)

        # validation loss가 가장 좋았던 모델 상태 저장
        if vloss < best_valid_loss:
            best_valid_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[{epoch:03d}] train_loss={mean_train_loss:.4f}  valid_loss={vloss:.4f}  valid_acc={vacc:.4f}")
    # 가장 좋은 상태로 모델 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        # best_state에서 복원된 모델을 파일로 저장
        torch.save(model.state_dict(), "best_model.pt")

    return model

# ---------- Main ----------

def main(args):
    # 현재 시각 문자열
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    # wandb에 기록할 설정값 딕셔너리다.
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [args.hidden, args.hidden],
        'activation': args.activation,
    }
    # W&B 초기화
    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="hw2-titanic",
        notes="Titanic classification (PyTorch MLP)",
        tags=["titanic", "classification", args.activation, f"bs{args.batch_size}"],
        name=current_time_str,
        config=config
    )

    # Data
    train_loader, valid_loader, input_dim = get_titanic_dataloaders(args.batch_size)

    model = MyModel(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        num_classes=2,
        activation=args.activation,
        dropout=0.1,
    )
    # 옵티마이저 세팅
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 실행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "/ input_dim:", input_dim, "/ activation:", args.activation, "/ batch:", args.batch_size)
    # 학습 루프 호출
    training_loop(model, optimizer, train_loader, valid_loader, max_epochs=args.epochs, device=device)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable W&B logging")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("-a", "--activation", type=str, default="relu", # 활성화 함수 인자 추가
                        choices=["relu", "sigmoid", "elu", "leaky_relu"],
                        help="Activation function (default: relu)")
    parser.add_argument("-H", "--hidden", type=int, default=64, help="Hidden units per layer (default: 64)")
    args = parser.parse_args()
    main(args)
