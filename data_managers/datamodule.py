from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import transforms

from data_managers.dataset import CustomDataset
from data_managers.transforms.transform_one import TransformOne


def collate_func(args):
    pass


class CustomDataModule(pl.LightningDataModule):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        # We hardcode dataset specific stuff here.
        self.transform = transforms.Compose([TransformOne(), ])

    def prepare_data(self):
        # download or pre-process data
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "train" or stage is None:
            full_dataset = CustomDataset(transform=self.transform)
            self.train, self.dev = random_split(
                full_dataset, [self.hparams.train_size, self.hparams.valid_size]
            )
            # If there are separate datasets
            # self.train = CustomDataset(**train_args, transform=self.transform)
            # self.val = CustomDataset(**val_args, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CustomDataset(transform=self.transform)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=collate_func,
            shuffle=True,
        )

    def val_dataloader(self):
        # RECOMMENDED
        return DataLoader(
            self.dev,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=collate_func,
            shuffle=False,
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=collate_func,
            shuffle=False,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # Dataset specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--data_dir", default="./", type=str)

        # training specific
        parser.add_argument("--train_size", default=55_000, type=int)
        parser.add_argument("--valid_size", default=5_000, type=int)
        parser.add_argument("--workers", default=8, type=int)

        return parser
