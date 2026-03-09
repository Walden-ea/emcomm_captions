# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch

from egg.core.util import move_to
from egg.core.callbacks import Callback


def compute_binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def compute_baseline_accuracy(num_dist, symbols, *dims):
    final = []
    for num_dim in dims:
        result = 0
        for j in range(num_dist + 1):
            probability = 1 / (j + 1)
            number_of_equal_dist = compute_binomial(num_dist, j)
            equal_dist = (1 / num_dim) ** (symbols * j)
            diff_dist = ((num_dim - 1) / num_dim) ** (symbols * (num_dist - j))
            result += probability * number_of_equal_dist * equal_dist * diff_dist
        final.append(result)

    return [round(elem, 4) * 100 for elem in final]


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(elems):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in elems:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy


def compute_mi_input_msgs(sender_inputs, messages):
    num_dimensions = len(sender_inputs[0])
    each_dim = [[] for _ in range(num_dimensions)]
    result = []
    for i, _ in enumerate(each_dim):
        for vector in sender_inputs:
            each_dim[i].append(vector[i])  # only works for 1-D sender inputs

    for i, dim_list in enumerate(each_dim):
        result.append(round(mutual_info(messages, dim_list), 4))

    print(
        f"| Entropy for each dimension of the input vectors = {[entropy(elem) for elem in each_dim]}"
    )
    print(f"| H(msg) = {entropy(messages)}")
    print(f"| MI = {result}")


def dump_sender_receiver(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    variable_length: bool,
    device: Optional[torch.device] = None,
):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether Gumbel-Softmax relaxation was used during training
    :param variable_length: whether variable-length communication is used
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training  # persist so we restore it back
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, messages, receiver_inputs, receiver_outputs = [], [], [], []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            # Under GS, the only output is a message; under Reinforce, two additional tensors are returned.
            # We don't need them.
            if not gs:
                message = message[0]

            output = game.receiver(message, receiver_input)
            if not gs:
                output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                sender_inputs.extend(zip(*sender_input))
            else:
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

            if gs:
                message = message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded

            if not variable_length:
                messages.extend(message)
                receiver_outputs.extend(output)
            else:
                # A trickier part is to handle EOS in the messages. It also might happen that not every message has EOS.
                # We cut messages at EOS if it is present or return the entire message otherwise. Note, EOS id is always
                # set to 0.

                for i in range(message.size(0)):
                    eos_positions = (message[i, :] == 0).nonzero()
                    message_end = (
                        eos_positions[0].item() if eos_positions.size(0) > 0 else -1
                    )
                    assert message_end == -1 or message[i, message_end] == 0
                    if message_end < 0:
                        messages.append(message[i, :])
                    else:
                        messages.append(message[i, : message_end + 1])

                    if gs:
                        receiver_outputs.append(output[i, message_end, ...])
                    else:
                        receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels

class DataRegeneratorCallback(Callback):
    """Callback to regenerate data iterators at the start of each epoch.
    
    This ensures new random distractors are sampled for each epoch when using
    a VectorsLoader-based data loader.
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def on_epoch_begin(self, epoch: int):
        """Regenerate train and validation data iterators at the start of each epoch."""
        # Set epoch if the loader supports it (e.g., PreGeneratedEpochCurriculumLoader)
        if hasattr(self.data_loader, 'set_epoch'):
            self.data_loader.set_epoch(epoch)
        
        train_data, validation_data, _ = self.data_loader.get_iterators()
        self.trainer.train_data = train_data
        self.trainer.validation_data = validation_data


class EpochDataLoaderCallback(Callback):
    """Callback to load pre-shuffled epoch-specific NPZ data files.
    
    Loads data from epoch-specific files following a template pattern at the start
    of each epoch. Useful for pre-generated, pre-shuffled datasets where each epoch
    has its own set of samples.
    """
    
    def __init__(self, data_loader, epoch_data_path_template):
        """
        Args:
            data_loader: A VectorsLoader instance used to process loaded data
            epoch_data_path_template: Path template with {epoch} placeholder, 
                                      e.g., '/path/data_3_distractors_{epoch}_epoch.npz'
        """
        self.data_loader = data_loader
        self.epoch_data_path_template = epoch_data_path_template
    
    def on_epoch_begin(self, epoch: int):
        """Load epoch-specific data file at the start of each epoch."""
        # Replace {epoch} placeholder in the template with the actual epoch number
        epoch_data_path = self.epoch_data_path_template.format(epoch=min((epoch//2)+5, 49))
        
        print(f"Loading epoch-specific data from: {epoch_data_path}")
        train_it, val_it, test_it = self.data_loader.get_iterators_load(epoch_data_path)

        self.trainer.train_data = train_it
        self.trainer.validation_data = val_it
        # # Load the NPZ file using the data_loader's load_data method
        # train, valid, test = self.data_loader.load_data(epoch_data_path)
        
        # # Convert loaded data to datasets and iterators
        # from src.objects_game.src.features import TupleDataset
        # from torch.utils import data
        
        # train_dataset = TupleDataset(*train)
        # valid_dataset = TupleDataset(*valid)
        # test_dataset = TupleDataset(*test)
        # print('SHAPE: ')
        # print(test_dataset[0][0].shape)
        # print(test_dataset[0][1].shape)
        
        # train_it = data.DataLoader(
        #     train_dataset,
        #     batch_size=self.data_loader.batch_size,
        #     shuffle=self.data_loader.shuffle_train_data,
        #     num_workers=0,
        # )
        
        # valid_it = data.DataLoader(
        #     valid_dataset,
        #     batch_size=self.data_loader.batch_size,
        #     shuffle=False,
        #     num_workers=0,
        # )
        
        # # test_it = data.DataLoader(
        # #     test_dataset,
        # #     batch_size=self.data_loader.batch_size,
        # #     shuffle=False,
        # #     num_workers=0,
        # # )
        
        # # Update trainer's data
        # self.trainer.train_data = train_it
        # self.trainer.validation_data = valid_it
