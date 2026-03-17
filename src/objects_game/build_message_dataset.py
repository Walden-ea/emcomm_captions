#!/usr/bin/env python3
"""
Build message datasets from a trained checkpoint.

This script generates messages for train, validation, and (optionally) test datasets
using a trained sender model. Messages are saved alongside the checkpoint in a 
messages subdirectory.

Usage:
    python build_message_dataset.py <config.yaml>
"""

import sys
import os
from pathlib import Path
from collections import namedtuple

import yaml
import torch
import torch.nn.functional as F
from datasets import load_from_disk, Dataset

from src.objects_game.src.archs import Sender, Receiver
from src.objects_game.src.features import VectorsLoader
import egg.core as core


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_best_checkpoint(checkpoint_dir, wandb_name):
    """Find the best checkpoint file in the checkpoint directory."""
    pattern_path = Path(checkpoint_dir) / wandb_name
    
    if not pattern_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {pattern_path}")
    
    # Look for best_epoch_*.pt files
    best_checkpoints = list(pattern_path.glob("best_epoch_*.pt"))
    
    if not best_checkpoints:
        raise FileNotFoundError(f"No best_epoch_*.pt files found in {pattern_path}")
    
    # Return the latest best checkpoint (highest epoch number)
    best_checkpoints.sort()
    return best_checkpoints[-1]


def init_game_from_checkpoint(checkpoint_path, device):
    """Initialize sender from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Reconstruct options from checkpoint
    # print(checkpoint['opts'])
    # for k in checkpoint['opts'].keys():
    #     c = k.lstrip('-').replace('-', '_')
    #     print(k, "->", c)
    # # clean_keys = [k.lstrip('-').replace('-', '_') for k in checkpoint['opts'].keys()]
    # clean_opts = {}
    # for k, v in checkpoint['opts'].items():
    #     key = k.lstrip('-').replace('-', '_')
    #     if key not in clean_opts:
    #         clean_opts[key] = v

    clean_opts = {k:v for k,v in checkpoint['opts'].items() if not k.startswith('--')}
    # clean_opts['sender_embedding'] = checkpoint['opts']['--sender_embedding']
    # clean_opts['reciever_embedding'] = checkpoint['opts']['--sender_embedding']
    # clean_opts['sender_hidden'] = checkpoint['opts']['--sender_embedding']
    # clean_opts['reciever_hidden'] = checkpoint['opts']['--sender_embedding']
    print(clean_opts)
    OptsNamedTuple = namedtuple('Opts', clean_opts.keys())
    opts = OptsNamedTuple(*clean_opts.values())
    
    # Initialize data loader to get number of features
    data_loader = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=False,#opts.shuffle_train_data,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=opts.load_data_path,
        seed=opts.data_seed,
    )
    
    print(f"Data loader initialized. Number of features: {data_loader.n_features}")
    
    # Create sender architecture
    sender_arch = Sender(n_features=data_loader.n_features, n_hidden=opts.sender_hidden)
    
    # Wrap with GS if needed
    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender_arch,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
    else:
        raise NotImplementedError(f"Unknown training mode: {opts.mode}")
    
    # Load checkpoint state
    sender.load_state_dict(checkpoint['sender'])
    sender.to(device)
    sender.eval()
    
    return sender, opts


def get_message_batch(sender, input_batch, device):
    """Generate messages for a batch of inputs."""
    with torch.no_grad():
        messages_probs = sender(torch.tensor(input_batch['features']).to(device))
        messages = messages_probs.argmax(dim=-1)
    
    return {'message': messages.cpu()}


def truncate_message(example):
    """Truncate message at first padding token (0)."""
    msg = example['message']
    
    # Find first padding token
    try:
        idx = msg.index(0)
        return msg[:idx]
    except ValueError:
        # No padding token found, return full message
        return msg


def validate_dataset_path(path_str, split_name):
    """Validate that a dataset path exists."""
    if not path_str:
        return None
    
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path for {split_name} not found: {path}\n"
            f"Please ensure the path is correct in the config file."
        )
    
    return path


def process_dataset(sender, dataset, split_name, device, batch_size=512):
    """Generate messages and truncate for a dataset split."""
    print(f"Generating messages for {split_name}...")
    ds_with_messages = dataset.map(
        lambda example: get_message_batch(sender, example, device),
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,  # Force recomputation, don't use cached results
    )
    
    print(f"Truncating messages for {split_name}...")
    ds_truncated = ds_with_messages.map(
        lambda example: {'message_truncated': truncate_message(example)},
        load_from_cache_file=False,  # Force recomputation, don't use cached results
    )
    
    return ds_truncated


def main(config_path):
    """Main function to build message dataset."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_dir = config.get('checkpoint_save_path', './checkpoints')
    wandb_name = config.get('wandb_name', 'default')
    
    checkpoint_path = find_best_checkpoint(checkpoint_dir, wandb_name)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Initialize sender from checkpoint
    sender, opts = init_game_from_checkpoint(checkpoint_path, device)
    print("Sender model loaded and initialized")
    
    # Create output directory for messages
    output_base_dir = Path(checkpoint_path).parent / "messages"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset paths from config
    datasets_to_process = {}
    
    train_path = config.get('dataset_train_path')
    if train_path:
        datasets_to_process['train'] = validate_dataset_path(train_path, 'train')
    else:
        raise ValueError("dataset_train_path is required in config")
    
    val_path = config.get('dataset_val_path')
    if val_path:
        datasets_to_process['val'] = validate_dataset_path(val_path, 'val')
    else:
        raise ValueError("dataset_val_path is required in config")
    
    test_path = config.get('dataset_test_path')
    if test_path:
        test_path_obj = validate_dataset_path(test_path, 'test')
        if test_path_obj:
            datasets_to_process['test'] = test_path_obj
    
    processed_splits = []
    
    for split, dataset_path in datasets_to_process.items():
        
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} dataset")
        print(f"{'='*60}")
        print(f"Loading {split} dataset from: {dataset_path}")
        
        ds = load_from_disk(str(dataset_path))
        print(f"{split.capitalize()} dataset loaded: {len(ds)} samples")
        
        # Process the dataset
        ds_truncated = process_dataset(sender, ds, split, device)
        print(f"✓ {split.capitalize()} messages generated and truncated")
        
        # Save the dataset
        output_path = output_base_dir / f"coco_{split}_message_captions_{opts.n_distractors}_distractors"
        print(f"Saving {split} dataset to: {output_path}")
        ds_truncated.save_to_disk(str(output_path))
        print(f"✓ {split.capitalize()} dataset saved successfully")
        
        processed_splits.append({
            'split': split,
            'path': output_path,
            'samples': len(ds_truncated)
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output base directory: {output_base_dir}")
    print(f"\nProcessed splits:")
    for info in processed_splits:
        print(f"  ✓ {info['split'].upper():6} - {info['samples']:6} samples -> {info['path'].name}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_message_dataset.py <config.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])
