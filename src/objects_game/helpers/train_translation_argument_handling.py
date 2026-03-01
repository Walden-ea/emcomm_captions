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


def _populate_parser(parser):
    """Factor out argument definitions so they can be reused."""

    # Dataset paths
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="../datasets/coco_train_msg_captions_5_distractors",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="../datasets/coco_val_msg_captions_5_distractors",
        help="Path to validation/test dataset",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default=None,
        help="Path to a dataset to use only for testing (only used with --test_only)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="best_model.pt",
        help="Path to save best model checkpoint or load from when testing",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="If set, skip training and run evaluation only on the provided test dataset",
    )

    # Model type selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="rnn",
        choices=["rnn", "transformer"],
        help="Model architecture: 'rnn' for LSTM or 'transformer' for Transformer",
    )

    # Model architecture
    parser.add_argument(
        "--src_vocab_size",
        type=int,
        default=71,
        help="Source vocabulary size (messages)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=512,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--enc_num_layers",
        type=int,
        default=2,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--dec_num_layers",
        type=int,
        default=2,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (for transformer model)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate",
    )
    parser.add_argument(
        "--pad_id",
        type=int,
        default=70,
        help="Padding ID for source sequences",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr_enc",
        type=float,
        default=1e-3,
        help="Encoder learning rate",
    )
    parser.add_argument(
        "--lr_dec",
        type=float,
        default=1e-3,
        help="Decoder learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=20,
        help="LR scheduler patience",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.9,
        help="LR scheduler decay factor",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EmComm-Caption-Translator",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="translation_baseline",
        help="Weights & Biases run name",
    )


def check_args(args):
    """Validate and process arguments."""
    pass
