"""
Hyperparameter optimization for translation model using Optuna.

Example usage:
    python hpo_translation.py --n_trials 20 --hpo_outdir ./hpo_results
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

from src.objects_game.train_translation import main, get_tokenizer_and_pad
from datasets import load_from_disk


def create_trial_args(trial, base_args):
    """Create training arguments with Optuna-suggested hyperparameters.
    
    Args:
        trial: Optuna trial object
        base_args: Base argparse Namespace with defaults
        
    Returns:
        Modified args with trial-suggested hyperparameters
    """
    # Learning rates (log scale)
    base_args.lr_enc = trial.suggest_float("lr_enc", 1e-5, 1e-2, log=True)
    base_args.lr_dec = trial.suggest_float("lr_dec", 1e-5, 1e-2, log=True)
    
    # Architecture dimensions
    base_args.emb_dim = trial.suggest_int("emb_dim", 64, 512, step=64)
    base_args.hid_dim = trial.suggest_int("hid_dim", 128, 1024, step=128)
    
    # Dropout (log scale) - more likely to sample smaller values
    base_args.dropout = trial.suggest_float("dropout", 0.0, 0.5, log=False)
    
    # Batch size (powers of 2)
    # base_args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    # Regularization / early stopping
    # base_args.patience = trial.suggest_int("patience", 5, 20)
    base_args.scheduler_patience = trial.suggest_int("scheduler_patience", 5, 30)
    base_args.scheduler_factor = trial.suggest_float("scheduler_factor", 0.5, 0.99)
    
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
    parser = argparse.ArgumentParser(description="Train translation model")
    parser.add_argument("--train_dataset_path", type=str, default="../datasets/coco_train_msg_captions")
    parser.add_argument("--val_dataset_path", type=str, default="../datasets/coco_val_msg_captions")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--src_vocab_size", type=int, default=71)
    parser.add_argument("--enc_num_layers", type=int, default=2)
    parser.add_argument("--dec_num_layers", type=int, default=2)
    parser.add_argument("--pad_id", type=int, default=70)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--wandb_project", type=str, default="EmComm-Caption-Translator")
    parser.add_argument("--wandb_name", type=str, default="")
    
    # Add defaults for hyperparameters to be tuned
    parser.add_argument("--lr_enc", type=float, default=1e-3)
    parser.add_argument("--lr_dec", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler_patience", type=int, default=7)
    parser.add_argument("--scheduler_factor", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=seed)
    
    args = parser.parse_args([])
    
    # Use trial to suggest hyperparameters
    args = create_trial_args(trial, args)
    
    # Create temp checkpoint for this trial
    with tempfile.TemporaryDirectory() as tmpdir:
        args.checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        args.wandb_name = f"trial_{trial.number}"
        
        try:
            # Train and return best validation loss
            best_val_loss = main(args)
            return best_val_loss
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
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
    pruner = MedianPruner(n_warmup_steps=5)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="translation_hpo"
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
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for translation model")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials to run")
    parser.add_argument("--hpo_outdir", type=str, default="./hpo_results",
                        help="Directory to save HPO results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    study = main_hpo(args)
