"""
This file runs the main training/val loop, etc. using Lightning Trainer.
"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from models.model import Model
from data_managers.datamodule import CustomDataModule


def main(args):
    # init modules
    dm = CustomDataModule(hparams=args)
    model = Model(hparams=args)

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args)

    dm.setup('train')
    trainer.fit(model, dm)


def main_cli():
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well
    seed_everything(123)
    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Model.add_model_specific_args(parser)
    # same goes for data modules
    parser = CustomDataModule.add_data_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    main_cli()
