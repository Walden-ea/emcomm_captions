# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import operator
import os, glob

# argument-handling helpers are located in a separate module
from src.objects_game.helpers.train_argument_handling import get_params

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from datasets import load_from_disk

import egg.core as core
from egg.core.util import move_to, _set_seed
from src.objects_game.src.archs import Receiver, Sender
from src.objects_game.src.features import VectorsLoader
from src.objects_game.src.trainers import Trainer
from src.objects_game.src.util import (
    compute_baseline_accuracy,
    compute_mi_input_msgs,
    dump_sender_receiver,
    entropy,
    mutual_info,
)



def loss(
    _sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input
):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {"acc": acc}

class BestAndLastCheckpoint(core.Callback):
    def __init__(self, path="checkpoints"):
        self.path = path
        self.best_loss = float("inf")
        os.makedirs(path, exist_ok=True)

    def _remove(self, pattern):
        for f in glob.glob(os.path.join(self.path, pattern)):
            os.remove(f)
    
    def _checkpoint(self, epoch):
        return {
            "epoch": epoch,
            "sender": self.trainer.game.sender.state_dict(),
            "receiver": self.trainer.game.receiver.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "opts": vars(self.trainer.opts),
            "common_opts": vars(self.trainer.common_opts),
        }

    def on_validation_end(self, loss, logs, epoch):
        # keep only one "last"
        self._remove("last_epoch_*.pt")
        torch.save(
            self._checkpoint(epoch),
            f"{self.path}/last_epoch_{epoch}.pt"
        )

        # keep only one "best"
        if loss < self.best_loss:
            self.best_loss = loss
            self._remove("best_epoch_*.pt")
            torch.save(
                self._checkpoint(epoch),
                f"{self.path}/best_epoch_{epoch}.pt"
            )

def main(params):
    opts = get_params(params)
    _set_seed(opts.random_seed)

    device = torch.device("cuda" if opts.cuda else "cpu")

    # Load datasets if using dataset-based tuple generation
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if opts.train_dataset_path:
        print(f"Loading train dataset from {opts.train_dataset_path}")
        train_dataset = load_from_disk(opts.train_dataset_path)
        
        if opts.val_dataset_path:
            print(f"Loading validation dataset from {opts.val_dataset_path}")
            val_dataset = load_from_disk(opts.val_dataset_path)
        
        if opts.test_dataset_path:
            print(f"Loading test dataset from {opts.test_dataset_path}")
            test_dataset = load_from_disk(opts.test_dataset_path)

    # Initialize data loader
    data_loader = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=opts.shuffle_train_data,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=opts.load_data_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        seed=opts.data_seed,
    )
    train_data, validation_data, test_data = data_loader.get_iterators()


    data_loader.upd_cl_options(opts)

    if opts.max_len > 1:
        baseline_msg = 'Cannot yet compute "smart" baseline value for messages of length greater than 1'
    else:
        baseline_msg = (
            f"\n| Baselines measures with {opts.n_distractors} distractors and messages of max_len = {opts.max_len}:\n"
            f"| Dummy random baseline: accuracy = {1 / (opts.n_distractors + 1)}\n"
        )
        if opts.perceptual_dimensions is not None and -1 not in opts.perceptual_dimensions:
            baseline_msg += f'| "Smart" baseline with perceptual_dimensions {opts.perceptual_dimensions} = {compute_baseline_accuracy(opts.n_distractors, opts.max_len, *opts.perceptual_dimensions)}\n'
        else:
            baseline_msg += f'| Data was loaded from an external file or dataset, thus no perceptual_dimension vector was provided, "smart baseline" cannot be computed\n'

    print(baseline_msg)

    sender = Sender(n_features=data_loader.n_features, n_hidden=opts.sender_hidden)

    receiver = Receiver(
        n_features=data_loader.n_features, linear_units=opts.receiver_hidden
    )

    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )

        receiver = core.RnnReceiverGS(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

        game = core.SenderReceiverRnnGS(sender, receiver, loss)
    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    optimizer = torch.optim.Adam(
        [
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ]
    )
    callbacks = [
        core.ConsoleLogger(as_json=True),
        BestAndLastCheckpoint(os.path.join(opts.checkpoint_save_path, opts.wandb_name)),
        ]#,  PlateauCallback()]
    if opts.mode.lower() == "gs":
        callbacks.append(core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1))
    
    # Track the best checkpoint callback to extract best validation loss
    best_checkpoint = None
    for callback in callbacks:
        if isinstance(callback, BestAndLastCheckpoint):
            best_checkpoint = callback
            break
    
    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
        opts=opts
    )
    trainer.train(n_epochs=opts.n_epochs)
    
    # Return best validation loss for HPO
    best_val_loss = best_checkpoint.best_loss if best_checkpoint else float('inf')

    if opts.evaluate:
        is_gs = opts.mode == "gs"
        (
            sender_inputs,
            messages,
            receiver_inputs,
            receiver_outputs,
            labels,
        ) = dump_sender_receiver(
            game, test_data, is_gs, variable_length=True, device=device
        )

        receiver_outputs = move_to(receiver_outputs, device)
        labels = move_to(labels, device)

        receiver_outputs = torch.stack(receiver_outputs)
        labels = torch.stack(labels)

        tensor_accuracy = receiver_outputs.argmax(dim=1) == labels
        accuracy = torch.mean(tensor_accuracy.float()).item()

        unique_dict = {}

        for elem in sender_inputs:
            target = ""
            for dim in elem:
                target += f"{str(int(dim.item()))}-"
            target = target[:-1]
            if target not in unique_dict:
                unique_dict[target] = True

        print(f"| Accuracy on test set: {accuracy}")

        # compute_mi_input_msgs(sender_inputs, messages)

        print(f"entropy sender inputs {entropy(sender_inputs)}")
        print(f"mi sender inputs msgs {mutual_info(sender_inputs, messages)}")
        # print('shape of sender inputs and messages:')
        # print(np.ndarray(sender_inputs).shape)

        if opts.dump_msg_folder:
            opts.dump_msg_folder.mkdir(exist_ok=True)
            msg_dict = {}

            output_msg = (
                f"data_path_{os.path.basename(opts.load_data_path)}_vocab_{opts.vocab_size}"
                f"_maxlen_{opts.max_len}_bsize_{opts.batch_size}"
                f"_n_distractors_{opts.n_distractors}_train_size_{opts.train_samples}"
                f"_valid_size_{opts.validation_samples}_test_size_{opts.test_samples}"
                f"_slr_{opts.sender_lr}_rlr_{opts.receiver_lr}_shidden_{opts.sender_hidden}"
                f"_rhidden_{opts.receiver_hidden}_semb_{opts.sender_embedding}"
                f"_remb_{opts.receiver_embedding}_mode_{opts.mode}"
                f"_scell_{opts.sender_cell}_rcell_{opts.receiver_cell}.msg"
            )

            output_file = opts.dump_msg_folder / output_msg
            with open(output_file, "w") as f:
                f.write(f"{opts}\n")
                for (
                    sender_input,
                    message,
                    receiver_input,
                    receiver_output,
                    label,
                ) in zip(
                    sender_inputs, messages, receiver_inputs, receiver_outputs, labels
                ):
                    sender_input = ",".join(map(str, sender_input.tolist()))
                    message = ",".join(map(str, message.tolist()))
                    distractors_list = receiver_input.tolist()
                    receiver_input = "; ".join(
                        [",".join(map(str, elem)) for elem in distractors_list]
                    )
                    if is_gs:
                        receiver_output = receiver_output.argmax()
                    f.write(
                        f"{sender_input} -> {receiver_input} -> {message} -> {receiver_output} (label={label.item()})\n"
                    )

                    if message in msg_dict:
                        msg_dict[message] += 1
                    else:
                        msg_dict[message] = 1

                sorted_msgs = sorted(
                    msg_dict.items(), key=operator.itemgetter(1), reverse=True
                )
                f.write(
                    f"\nUnique target vectors seen by sender: {len(unique_dict.keys())}\n"
                )
                f.write(f"Unique messages produced by sender: {len(msg_dict.keys())}\n")
                f.write(f"Messagses: 'msg' : msg_count: {str(sorted_msgs)}\n")
                f.write(f"\nAccuracy: {accuracy}")
    
    return best_val_loss


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    # forward the single path as a list so get_params can handle it
    main([sys.argv[1]])
