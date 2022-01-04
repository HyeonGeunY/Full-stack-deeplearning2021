"""
Experiment-running framework
"""

import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from text_recognizer import lit_models

np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """
    Import class from a module, e.g, 'text_recognizer.modules.MLP'
    """

    module_name, class_name = module_and_class_name.rsplit(
        ".", 1
    )  # rsplit : "." 기준으로 오른쪽부터 두번째 인자 숫자 만큼 나눔 if 0 -> 나누지 않음.

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_


def _setup_parser():
    """data, model, trainer, etc 관련 argument에 대한 파이썬 Argumentparser setup"""

    parser = argparse.ArgumentParser(add_help=False)
    # --max_epochs, --gpus, --precision과 같은 trainer argument
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[
        1
    ].title = "Trainer Args"  # optional argument의 titel을 Trainer Args로 바꿈 [0] : postioner, [1]: optioner, [2]: pl.Trainer (pl에서 생성한 flags)
    parser = argparse.ArgumentParser(
        add_help=False, parents=[trainer_parser]
    )  # pl.Trainer argparser의 모든 flag를 가지고 있는 parser 생성

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()  # 현재까지 값을 가지고 있는 arg (temp_args)와 없는 arg(_)분리
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser


def main():
    """
    Run an experiment

    Sample command
    '''
    python training/run_experiment.py --max_epochs=3 --gpus='0' --num_workers=20 --model_class=MLP --data_class=MNIST
    '''
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:  # 기존 model를 load해서 사용할 떄
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epochs:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    # filename
    # save any arbitrary metrics like `val_loss`, etc. in name
    # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
    # >>> checkpoint_callback = ModelCheckpoint(
    # ...     dirpath='my/path',
    # ...     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    # ... )

    # pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint(dirpath=None, filenam= ..
    # By default, dirpath is None and will be set at runtime to the location specified by Trainer’s default_root_dir or weights_save_path arguments, and if the Trainer uses a logger, the path will also contain logger name and version.

    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weight_summary = "full"  # model의 전체 summary 출력

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs"
    )
    # argparse 로 전달해도 되고, callback 등 인자로 전달해도 된다.

    # 각 trainer method에 맞춰서 train_loader, valid_loader, test_loader를 넣어주어도 되지만
    # data 모듈을 만들어서 넣어줘도 된다.

    # tune : Runs routines to tune hyperparameters before training.
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
