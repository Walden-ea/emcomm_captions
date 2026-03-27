import os
import pathlib

import wandb
from typing import List, Optional


import torch
from torch.utils.data import DataLoader

# from .batch import Batch
from egg.core.callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
    WandbLogger as CoreWandbLogger,
)


import argparse
import json
import os
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

import torch
import wandb
from rich.columns import Columns
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from egg.core.interaction import Interaction
from egg.core.util import get_summary_writer


# from .distributed import get_preemptive_checkpoint_dir
# from .interaction import Interaction
from egg.core.util import get_opts, move_to
from egg.core.trainers import Trainer as CoreTrainer

class WandbLogger(CoreWandbLogger):
    def __init__(
        self,
        opts: Union[argparse.ArgumentParser, Dict, str, None] = None,
        project: Optional[str] = 'EmComm-Caption',
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        log_every: int = 50,
        **kwargs,
        
    ):
        self.opts = opts
        self.log_every = log_every
        wandb.init(project=project, id=run_id, name=run_name, **kwargs)
        wandb.config.update(opts)

    def _log_metrics(self, phase: str, loss: float, logs: Interaction, epoch: float = None, log_loss: bool = True):
        """Helper for logging losses and auxiliary metrics."""
        metrics = {f"{phase}/loss": loss} if log_loss else {}
        if epoch is not None:
            metrics["epoch"] = epoch
        for k, v in logs.aux.items():
            metrics[f"{phase}/{k}"] = v.mean()
        self.log_to_wandb(metrics)
    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training and self.trainer.distributed_context.is_leader:
            if batch_id % self.log_every == 0:
                self._log_metrics("train", loss, logs)
            self.log_to_wandb({"batch_loss": loss}, commit=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self._log_metrics("test", loss, logs, epoch)
    
    def on_validation_end_tagged(self, loss: float, logs: Interaction, epoch: int, tag='default'):
        if self.trainer.distributed_context.is_leader:
            self._log_metrics(f"test/{tag}", loss, logs, epoch, log_loss=False)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self._log_metrics("train", loss, logs, epoch)

    def log_train_hyperparams(self, hyperparam_dict: Dict[str, float], epoch: int):
            if self.trainer.distributed_context.is_leader:
                metrics = {f'train/{k}':v for k,v in hyperparam_dict.items()}
                metrics["epoch"] = epoch
                self.log_to_wandb(metrics)


class Trainer(CoreTrainer):
    """
        added wandb and whatever else would be needed
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        opts,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        additional_validation_splits: Optional[Dict[str, DataLoader]] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        super().__init__(
            game,
            optimizer,
            train_data,
            optimizer_scheduler,
            validation_data,
            device,
            callbacks,
            grad_norm,
            aggregate_interaction_logs,
        )

        common_opts = get_opts()
        self.opts = opts
        self.common_opts = common_opts
        self.additional_validation_splits = additional_validation_splits
        
        if self.distributed_context.is_leader and common_opts.wandb:
            # assert (
            #     common_opts.tensorboard_dir
            # ), "tensorboard directory has to be specified"
            wandb_logger = WandbLogger(opts, run_name=opts.wandb_name)
            self.callbacks.append(wandb_logger)
    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)

            train_loss, train_interaction = self.train_epoch()

            self.log_train_hyperparams(epoch)
            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_interaction, epoch + 1)

            validation_loss, validation_interaction = self.validate(epoch)

            if self.should_stop:
                for callback in self.callbacks:
                    callback.on_early_stopping(
                        train_loss,
                        train_interaction,
                        epoch + 1,
                        validation_loss,
                        validation_interaction,
                    )
                break

        for callback in self.callbacks:
            callback.on_train_end()
    def validate(self, epoch):
        validation_loss = validation_interaction = None
        if (
            self.validation_data is not None
            and self.validation_freq > 0
            and (epoch + 1) % self.validation_freq == 0
        ):
            for callback in self.callbacks:
                callback.on_validation_begin(epoch + 1)
            validation_loss, validation_interaction = self.eval()

            for callback in self.callbacks:
                callback.on_validation_end(
                    validation_loss, validation_interaction, epoch + 1
                )
            self.validate_on_additional_splits(epoch)
        return validation_loss, validation_interaction
    
    def validate_on_additional_splits(self, epoch):
        for tag, split in self.additional_validation_splits.items():
            validation_loss, validation_interaction = self.eval(split)
            try:
                wandb_cb = next(
                    cb for cb in self.callbacks if isinstance(cb, WandbLogger)
                )
            except:
                print('No WANDB callback found!')
                return

            wandb_cb.on_validation_end_tagged(
                validation_loss, validation_interaction, epoch + 1, tag
            )
    def log_train_hyperparams(self, epoch):
        try:
            wandb_cb = next(
                cb for cb in self.callbacks if isinstance(cb, WandbLogger)
            )
        except:
            print('No WANDB callback found!')
            return
        wandb_cb.log_train_hyperparams({'temperature': self.game.sender.temperature}, epoch)
