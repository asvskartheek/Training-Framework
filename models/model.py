from argparse import ArgumentParser

import pytorch_lightning as pl
from torch import optim


class Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        # AdamW is my default choice
        return optim.AdamW(self.parameters())

    def get_loss(self, output, target):
        pass

    def training_step(self, batch, batch_idx):
        input = batch.input
        target = batch.label

        output = self(input, target)

        loss = self.get_loss(output, target)

        result = pl.TrainResult(loss)
        result.loss = loss
        result.log('train_loss', loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        input = batch.input
        target = batch.label

        output = self(input, target)

        loss = self.get_loss(output, target)
        pl.EvalResult()

        result = pl.EvalResult()
        result.val_loss = loss

        return result

    def test_step(self, batch, batch_idx):
        input = batch.input
        target = batch.label

        output = self(input, target)
        loss = self.get_loss(output, target)

        result = pl.EvalResult()
        result.test_loss = loss

        return result

    def validation_epoch_end(self, validation_step_results):
        loss = validation_step_results.val_loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result

    def test_epoch_end(self, test_step_results):
        loss = test_step_results.test_loss.mean()

        result = pl.EvalResult()
        result.log('test_loss', loss, prog_bar=True)
        return result

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # Model specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add sth

        return parser