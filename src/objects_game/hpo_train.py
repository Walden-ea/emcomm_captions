"""
Hyperparameter optimization for objects game training using Optuna.

Example usage:
    python hpo_train.py --n_trials 20 --hpo_outdir ./hpo_results
    python hpo_train.py --n_trials 30 --hpo_outdir ./hpo_results_long
"""

import argparse
import logging
import numpy as np
import os
import random
import tempfile
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from src.objects_game.train import main, get_params


def create_trial_args(trial, base_args):
    """Create training arguments with Optuna-suggested hyperparameters.
    
    Args:
        trial: Optuna trial object
        base_args: Base argparse Namespace with defaults
        
    Returns:
        Modified args with trial-suggested hyperparameters
    """
    # Learning rates (log scale)
    base_args.sender_lr = trial.suggest_float("sender_lr", 1e-4, 1e-1, log=True)
    base_args.receiver_lr = trial.suggest_float("receiver_lr", 1e-4, 1e-1, log=True)
    
    # Architecture dimensions
    base_args.sender_hidden = trial.suggest_int("sender_hidden", 32, 256, step=16)
    base_args.receiver_hidden = trial.suggest_int("receiver_hidden", 32, 256, step=16)
    base_args.sender_embedding = trial.suggest_int("sender_embedding", 8, 128, step=8)
    base_args.receiver_embedding = trial.suggest_int("receiver_embedding", 8, 128, step=8)
    
    # Temperature for Gumbel-Softmax
    base_args.temperature = trial.suggest_float("temperature", 0.5, 2.0, log=False)
    
    # Batch size (powers of 2)
    # base_args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    return base_args


def objective(trial):
    """Optuna objective function for hyperparameter search."""
    
    # Set random seeds for reproducibility (passed via global or use trial seed for variation)
    seed = 42 + trial.number  # Each trial gets a different seed for diversity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Parse base args
    parser = argparse.ArgumentParser(description="Train objects game")
    
    # Data arguments
    parser.add_argument(
        "--perceptual_dimensions",
        type=str,
        default="[4, 4, 4, 4, 4]",
        help="Number of features for every perceptual dimension",
    )
    parser.add_argument(
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
        default=1e4,
        help="Number of tuples in training data",
    )
    parser.add_argument(
        "--validation_samples",
        type=float,
        default=1e3,
        help="Number of tuples in validation data",
    )
    parser.add_argument(
        "--test_samples",
        type=float,
        default=1e3,
        help="Number of tuples in test data",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=111,
        help="Seed for random creation of train, validation and test tuples",
    )
    parser.add_argument(
        "--shuffle_train_data",
        action="store_true",
        default=False,
        help="Shuffle train data before every epoch",
    )
    
    # Architecture arguments
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver",
    )
    
    # Cell types
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm}",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm}",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Sender's parameters",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Receiver's parameters",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    
    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether GumbelSoftmax relaxation is used for training",
    )
    
    # Output arguments
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
    
    # Logging arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Logging to wandb",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="hpo_train",
        help="Optional wandb run name",
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run with pdb enabled",
    )
    
    # EGG core arguments (required but not tuned)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=70,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Max message length",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA for training",
    )
    
    args = parser.parse_args([])
    
    # Use trial to suggest hyperparameters
    args = create_trial_args(trial, args)
    
    # Set wandb name for tracking trials
    args.wandb_name = f"trial_{trial.number}"
    
    try:
        # Train and return best validation loss
        best_val_loss = main(args)
        return best_val_loss
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.TrialPruned()


def main_hpo(args):
    """Run hyperparameter optimization."""
    
    # Create output directory
    os.makedirs(args.hpo_outdir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.hpo_outdir, 'hpo.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting HPO with {args.n_trials} trials")
    logger.info(f"Results will be saved to {args.hpo_outdir}")
    
    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=2)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="objects_game_hpo"
    )
    
    # Optimize
    print(f"num trials: {args.n_trials}")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    
    # Save results
    logger.info("Optimization complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best loss: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, val in study.best_params.items():
        logger.info(f"  {key}: {val}")
    
    # Save study to file
    study.trials_dataframe().to_csv(
        os.path.join(args.hpo_outdir, "trials.csv"), 
        index=False
    )
    
    # Save best params to file
    with open(os.path.join(args.hpo_outdir, "best_params.txt"), "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best loss: {study.best_value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for key, val in study.best_params.items():
            f.write(f"  {key}: {val}\n")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for objects game training")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials to run")
    parser.add_argument("--hpo_outdir", type=str, default="./hpo_results_protocol",
                        help="Directory to save HPO results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    study = main_hpo(args)
