# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import pathlib

import egg.core as core

from src.objects_game import yaml_config


def get_params(params):
    """Return an argparse.Namespace containing training options.

    ``params`` may be one of the following:

    * a list/tuple of strings: interpreted as command line tokens and
      parsed exactly as before.  This is what HPO code still passes.
    * a single string ending in ``.yaml``/``.yml``: the file is
      loaded and its contents used to populate a namespace.  Missing
      values are filled with the defaults that the argument parser
      defines.
    * an ``argparse.Namespace`` instance: returned unchanged.
    """

    # allow callers to pass through an already-constructed namespace
    if isinstance(params, argparse.Namespace):
        check_args(params)
        return params

    # if the caller gave us a yaml path, load it and merge with defaults
    if isinstance(params, (list, tuple)) and len(params) == 1:
        candidate = params[0]
        if isinstance(candidate, str) and pathlib.Path(candidate).suffix in (
            ".yaml", ".yml"
        ):
            yaml_args = yaml_config.load_yaml_config(candidate)
            # build a parser just to obtain defaults and then overwrite
            # them with whatever was present in the yaml file
            parser = argparse.ArgumentParser()
            # re‑use the same argument definitions as before
            _populate_parser(parser)
            default_args = core.init(parser, [])
            for k, v in vars(yaml_args).items():
                setattr(default_args, k, v)
            check_args(default_args)
            print(default_args)
            return default_args

    # fall back to original command-line parsing behavior
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = core.init(parser, params)
    check_args(args)
    print(args)
    return args


# The three helpers below were formerly defined in ``train.py``.  They are
# kept together because they form the low-level argument handling logic that
# is otherwise orthogonal to the rest of the training script.

def _populate_parser(parser):
    """Factor out argument definitions so they can be reused."""

    input_data = parser.add_mutually_exclusive_group()
    input_data.add_argument(
        "--perceptual_dimensions",
        type=str,
        default="[4, 4, 4, 4, 4]",
        help="Number of features for every perceptual dimension",
    )
    input_data.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path to .npz data file to load",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=3,
        help="Number of distractor objects for the receiver (default: 3)",
    )
    parser.add_argument(
        "--train_samples",
        type=float,
        default=1e5,
        help="Number of tuples in training data (default: 1e6)",
    )
    parser.add_argument(
        "--validation_samples",
        type=float,
        default=1e3,
        help="Number of tuples in validation data (default: 1e4)",
    )
    parser.add_argument(
        "--test_samples",
        type=float,
        default=1e3,
        help="Number of tuples in test data (default: 1e3)",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=111,
        help="Seed for random creation of train, validation and test tuples (default: 111)",
    )
    parser.add_argument(
        "--shuffle_train_data",
        action="store_true",
        default=False,
        help="Shuffle train data before every epoch (default: False)",
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 50)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 50)",
    )

    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Sender's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Receiver's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {gs only at the moment}"
        "(default: rf)",
    )

    parser.add_argument(
        "--output_json",
        action="store_true",
        default=False,
        help="If set, egg will output validation stats in json format (default: False)",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate trained model on test data",
    )

    parser.add_argument(
        "--dump_data_folder",
        type=str,
        default=None,
        help="Folder where file with dumped data will be created",
    )
    parser.add_argument(
        "--dump_msg_folder",
        type=str,
        default=None,
        help="Folder where file with dumped messages will be created",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run egg/objects_game with pdb enabled",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Logging to wandb",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        help="Optional wandb run name",
    )
    parser.add_argument(
        "--checkpoint_save_path",
        type=str,
        default="checkpoints",
        help="Path where checkpoints will be saved (default: checkpoints)",
    )


def check_args(args):
    args.train_samples, args.validation_samples, args.test_samples = (
        int(args.train_samples),
        int(args.validation_samples),
        int(args.test_samples),
    )

    try:
        args.perceptual_dimensions = eval(args.perceptual_dimensions)
    except SyntaxError:
        print(
            "The format of the # of perceptual dimensions param is not correct. Please change it to string representing a list of int. Correct format: '[int, ..., int]' "
        )
        exit(1)

    if args.debug:
        import pdb

        pdb.set_trace()

    args.n_features = len(args.perceptual_dimensions)

    # can't set data loading and data dumping at the same time
    assert not (
        args.load_data_path and args.dump_data_folder
    ), "Cannot set folder to dump data while setting path to vectors to be loaded. Are you trying to dump the same vectors that you are loading?"

    args.dump_msg_folder = (
        pathlib.Path(args.dump_msg_folder) if args.dump_msg_folder is not None else None
    )

    if (not args.evaluate) and args.dump_msg_folder:
        print(
            "| WARNING --dump_msg_folder was set without --evaluate. Evaluation will not be performed nor any results will be dumped. Please set --evaluate"
        )
