from __future__ import print_function

import operator
import os
import sys
import glob
from pathlib import Path
from collections import namedtuple

import yaml
import torch
import torch.nn.functional as F
import torch.utils.data

from datasets import load_from_disk

import egg.core as core
from egg.core.util import move_to, _set_seed
from src.objects_game.src.archs import Receiver, Sender
from src.objects_game.src.features import VectorsLoader
from src.objects_game.src.features_extended import MultiSplitVectorsLoader
from src.objects_game.src.util import (
    dump_sender_receiver,
    entropy,
    mutual_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_checkpoint(checkpoint_dir: str, wandb_name: str, prefer: str = "best"):
    """Return path to best (or last) checkpoint."""
    base = Path(checkpoint_dir) / wandb_name
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base}")

    pattern = f"{prefer}_epoch_*.pt"
    hits = sorted(base.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No {pattern} files found in {base}")

    return hits[-1]   # highest epoch number


def build_game_from_checkpoint(checkpoint_path, device):
    """Reconstruct the full sender+receiver game from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Rebuild opts (strip any '--' prefixed duplicate keys)
    clean_opts = {k: v for k, v in checkpoint["opts"].items()
                  if not k.startswith("--")}
    OptsNT = namedtuple("Opts", clean_opts.keys())
    opts = OptsNT(*clean_opts.values())

    print(f"Checkpoint opts:\n{clean_opts}\n")

    # Need n_features – spin up a throwaway data loader
    data_loader = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=False,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=opts.load_data_path,
        seed=opts.data_seed,
    )
    n_features = data_loader.n_features
    print(f"n_features = {n_features}")

    # Build architectures
    sender_arch = Sender(n_features=n_features, n_hidden=opts.sender_hidden)
    receiver_arch = Receiver(n_features=n_features, linear_units=opts.receiver_hidden)

    if opts.mode.lower() != "gs":
        raise NotImplementedError(f"Unknown training mode: {opts.mode}")

    sender = core.RnnSenderGS(
        sender_arch,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        temperature=opts.temperature,
    )
    receiver = core.RnnReceiverGS(
        receiver_arch,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
    )

    def loss(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
        acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
        loss_val = F.cross_entropy(receiver_output, _labels, reduction="none")
        return loss_val, {"acc": acc}

    game = core.SenderReceiverRnnGS(sender, receiver, loss)

    # Load weights
    game.sender.load_state_dict(checkpoint["sender"])
    game.receiver.load_state_dict(checkpoint["receiver"])
    game.to(device)
    game.eval()

    return game, opts, data_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Find & load checkpoint ----------------------------------------
    checkpoint_dir = config.get("checkpoint_save_path", "./checkpoints")
    wandb_name = config.get("wandb_name", "default")
    prefer = config.get("eval_checkpoint", "best")   # "best" or "last"

    checkpoint_path = find_checkpoint(checkpoint_dir, wandb_name, prefer=prefer)
    print(f"Loading checkpoint: {checkpoint_path}\n")

    game, opts, data_loader = build_game_from_checkpoint(checkpoint_path, device)

    # ---- Build test iterator -------------------------------------------
    hard_test_data_path = "/home/elena/emcomm/emcomm_captions/epoch_datasets_slower/hard_test_data_dummy_train_3_distractors_1500_epoch.npz"
    data_loader_test = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=False,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=hard_test_data_path,
        seed=opts.data_seed,
    )
    _, _, test_data = data_loader_test.get_iterators()

    # Optional: combined val splits
    val_datasets_path = r"/home/elena/emcomm/emcomm_captions/combined_val/data_3_distractors_combined_with_noise_val.npz"
    if os.path.exists(val_datasets_path):
        val_loader = MultiSplitVectorsLoader(
            perceptual_dimensions=opts.perceptual_dimensions,
            n_distractors=opts.n_distractors,
            batch_size=opts.batch_size,
            train_samples=opts.train_samples,
            validation_samples=opts.validation_samples,
            test_samples=opts.test_samples,
            shuffle_train_data=False,
            dump_data_folder=opts.dump_data_folder,
            load_data_path=val_datasets_path,
            seed=opts.data_seed,
        )
        val_iters_dict = val_loader.get_iterators()
        print("Available val splits:", list(val_iters_dict.keys()))
    else:
        val_iters_dict = {}
        print("Skipping multi-split val (path not found or not set).")

    # ---- Evaluate -------------------------------------------------------
    is_gs = opts.mode.lower() == "gs"

    def evaluate_split(split_name, data_iter):
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name}")
        print(f"{'='*60}")

        sender_inputs, messages, receiver_inputs, receiver_outputs, labels = \
            dump_sender_receiver(game, data_iter, is_gs, variable_length=True, device=device)

        receiver_outputs = move_to(receiver_outputs, device)
        labels = move_to(labels, device)

        receiver_outputs = torch.stack(receiver_outputs)
        labels = torch.stack(labels)

        tensor_accuracy = receiver_outputs.argmax(dim=1) == labels
        accuracy = torch.mean(tensor_accuracy.float()).item()

        ent = entropy(sender_inputs)
        mi = mutual_info(sender_inputs, messages)

        print(f"  Accuracy          : {accuracy:.4f}")
        print(f"  Entropy (inputs)  : {ent:.4f}")
        print(f"  MI (inputs, msgs) : {mi:.4f}")

        return {"accuracy": accuracy, "entropy": ent, "mi": mi}

    results = {}

    # Hard test set
    results["hard_test"] = evaluate_split("hard_test", test_data)

    # All val splits
    for split_name, split_iter in val_iters_dict.items():
        results[split_name] = evaluate_split(split_name, split_iter)

    # ---- Summary --------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"{'Split':<25} {'Accuracy':>10} {'Entropy':>10} {'MI':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['entropy']:>10.4f} {r['mi']:>10.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])