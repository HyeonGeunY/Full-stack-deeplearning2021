import argparse
import pytorch_lightning as pl
import torch
import torchmetrics

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(torchmetrics.Accuracy):
    """
    Accuracy metrics with a hack
    """

    # def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
    #     """
    #     pytorchlightning 1.2+ 기준 probability 값이 0~1 범위를 벗어나는 버그 존재

    #     preds를 받기 전에 normalize를 해주어 해결
    #         - 예측 결과는 max 확률이 중요함으로 값의 순위만 바뀌지 않는다면 문제 x
    #     """

    #     if preds.min() < 0 or preds.max() > 1:
    #         preds = torch.nn.functional.softmax(preds, dim=-1)
    #     super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):
    """
    파이토치 모듈로 initialized 되는 일반벅인 pytorch-lightning class
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model  # 모델을 받음
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        # getattr(object, name[, default]), object에서 "name" attribute를 가져온다. 값이 없을 경우 default값을 가져온다.
        self.optimizer_class = getattr(
            torch.optim, optimizer
        )  # torch.optim에서 optimzer attribute을 가져온다.

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim"
        )
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument(
            "--loss", type=str, default=LOSS, help="loss function from torch.nn.functional"
        )

        return parser

    def configure_optimizers(self):
        """
        설정 값으로 optimizer를 구성하여 반환

        returns
        ----------
        if onecycle learning rate
            dict{optimizer, learnrate schduler, monitor}
                monitor : 감독할 loss 종류 (str)
        else
            optimizer
        """
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            # one_cycle 정책을 사용하지 않는다면 고정된 lr을 갖는 optimizer 반환
            return optimizer

        scheduler = torch.optim.lr_scheduler.OnecycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # forward 매소드 호출
        loss = self.loss_fn(logits, y.to(dtype=torch.long))
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.to(dtype=torch.long))
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
