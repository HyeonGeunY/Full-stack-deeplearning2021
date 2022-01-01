import argparse
import pytorch_lightning as pl
import torch

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(pl.metrics.Accuracy):
    """
    Accuracy metrics with a hack
    """

    def update(self, preds: torch.Torch, target: torch.Tensor) -> None:
        """
        pytorchlightning 1.2+ 기준 probability 값이 0~1 범위를 벗어나는 버그 존재

        preds를 받기 전에 normalize를 해주어 해결
            - 예측 결과는 max 확률이 중요함으로 값의 순위만 바뀌지 않는다면 문제 x
        """

        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.lightningModule):
    """ """
