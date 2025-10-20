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

# try:
#     from torch.cuda.amp import GradScaler, autocast
# except ImportError:
#     pass
from egg.core.trainers import Trainer as CoreTrainer

class WandbLogger(CoreWandbLogger):
    def __init__(
        self,
        opts: Union[argparse.ArgumentParser, Dict, str, None] = None,
        project: Optional[str] = 'EmComm-Caption',
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        **kwargs,
    ):
        self.opts = opts

        wandb.init(project=project, id=run_id, name=run_name, **kwargs)
        wandb.config.update(opts)

    def _log_metrics(self, phase: str, loss: float, logs: Interaction, epoch: int):
        """Helper for logging losses and auxiliary metrics."""
        metrics = {f"{phase}/loss": loss, "epoch": epoch}
        for k, v in logs.aux.items():
            metrics[f"{phase}/{k}"] = v.mean()
        self.log_to_wandb(metrics)#, commit=True)
        # self.log_to_wandb({
        #     f"{phase}/interaction/message": {logs.message},
        #     f"{phase}/interaction/receiver_output": {logs.receiver_output},
        #     }, commit=True)
        # print({
        #     f"{phase}/interaction/message": {logs.message},
        #     f"{phase}/interaction/receiver_output": {logs.receiver_output},
        #     })

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self._log_metrics("test", loss, logs, epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self._log_metrics("train", loss, logs, epoch)


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

        if self.distributed_context.is_leader and common_opts.wandb:
            # assert (
            #     common_opts.tensorboard_dir
            # ), "tensorboard directory has to be specified"
            wandb_logger = WandbLogger(opts, run_name=opts.wandb_name)
            self.callbacks.append(wandb_logger)
