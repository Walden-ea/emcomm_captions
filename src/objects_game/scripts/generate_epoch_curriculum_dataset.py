"""
Script to generate epoch-specific curriculum learning datasets.

This script creates separate npz files for each epoch, where each epoch contains
tuple datasets with different distractor difficulty levels based on curriculum learning.
Earlier epochs have easier (more random) distractors, while later epochs have 
harder (more similar) distractors.

The curriculum strategy samples n_distractors + epoch candidates by cosine similarity,
then randomly selects n_distractors from those candidates.

Usage:
python generate_epoch_curriculum_dataset.py \
    --train_dataset_path path/to/train/dataset \
    --val_dataset_path path/to/val/dataset \
    --test_dataset_path path/to/test/dataset \
    --output_dir path/to/output \
    --n_distractors 3 \
    --n_epochs 100 \
    --seed 42
"""

import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk


def load_dataset_features(dataset_path, feature_column="features"):
    """Load feature vectors from HuggingFace dataset.
    
    Args:
        dataset_path: Path to HuggingFace dataset directory
        feature_column: Name of the column containing features (default: "features")
        
    Returns:
        numpy array of shape (n_samples, n_features)
    """
    dataset = load_from_disk(dataset_path)
    features = np.array(dataset[feature_column])
    print(f"  Loaded {len(features)} samples of shape {features.shape[1:]}")
    return features


def create_curriculum_tuples_for_epoch(
    all_vectors,
    n_samples,
    n_distractors,
    epoch,
    random_state,
    use_similarity=True
):
    """Create tuples with curriculum learning for a specific epoch.
    
    Uses cosine similarity to select distractors. The number of candidate
    distractors is n_distractors + epoch. To avoid computing similarity on
    all vectors (which is too expensive for large datasets), we first sample
    n_candidates from the available vectors, then compute similarity only
    on those sampled candidates, and select the closest ones.
    
    Args:
        all_vectors: numpy array of shape (n_samples, n_features)
        n_samples: number of tuples to create
        n_distractors: base number of distractors per tuple
        epoch: current epoch (affects difficulty)
        random_state: numpy RandomState for reproducibility
        use_similarity: whether to use cosine similarity (True) or random (False)
        
    Returns:
        tuple of (tuples, labels) where tuples has shape (n_samples, n_distractors+1, n_features)
    """
    n_objects = len(all_vectors)
    tuple_dim = n_distractors + 1
    
    # Pre-allocate output array
    tuples = np.zeros(
        (n_samples, tuple_dim, all_vectors.shape[1]), 
        dtype=all_vectors.dtype
    )
    
    # Assign random target positions for each sample
    target_idxs = random_state.randint(0, tuple_dim, n_samples)
    
    # Place all targets at once
    tuples[np.arange(n_samples), target_idxs] = all_vectors[:n_samples]
    
    # Fill distractor positions
    for target_idx in tqdm(range(n_samples), desc=f"  Creating tuples for epoch {epoch}"):
        target_pos = target_idxs[target_idx]
        
        # Get non-target indices
        non_target_indices = np.concatenate([
            np.arange(target_idx),
            np.arange(target_idx + 1, n_objects)
        ])
        
        if use_similarity:
            # Number of candidates increases with epoch: basic distractors + extra similar ones
            n_candidates = min(
                n_distractors + epoch*2,
                len(non_target_indices)
            )
            
            # Sample candidate indices from non-target vectors
            sampled_candidate_positions = random_state.choice(
                np.arange(len(non_target_indices)),
                size=n_candidates,
                replace=False
            )
            sampled_candidate_indices = non_target_indices[sampled_candidate_positions]
            
            # Compute cosine similarity only between target and sampled candidates
            target_vector = all_vectors[target_idx:target_idx+1]
            candidate_vectors = all_vectors[sampled_candidate_indices]
            similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
            
            # Select the n_distractors closest vectors by cosine similarity from sampled candidates
            closest_positions = np.argsort(-similarities)[:n_distractors]
            distractor_indices = sampled_candidate_indices[closest_positions]
        else:
            # Random selection (fallback)
            distractor_indices = random_state.choice(
                non_target_indices,
                size=n_distractors,
                replace=False
            )
        
        # Get positions to fill (all except target position)
        fill_positions = [i for i in range(tuple_dim) if i != target_pos]
        
        # Place all distractors
        tuples[target_idx, fill_positions] = all_vectors[distractor_indices]
    
    return tuples, target_idxs


def generate_epoch_datasets(
    train_features,
    val_features,
    test_features,
    output_dir,
    n_distractors=3,
    n_epochs=100,
    seed=42,
    use_similarity=True
):
    """Generate epoch-specific curriculum datasets.
    
    Creates a folder structure with epoch-specific npz files:
    output_dir/
        epoch_0/
            data.npz
        epoch_1/
            ...
        ...
    
    The number of tuples per split equals the size of the input features.
    
    Args:
        train_features: numpy array of training feature vectors (n_train, n_features)
        val_features: numpy array of validation feature vectors (n_val, n_features)
        test_features: numpy array of test feature vectors (n_test, n_features)
        output_dir: directory to save epoch datasets
        n_distractors: number of distractors per tuple
        n_epochs: number of epochs to generate
        seed: random seed for reproducibility
        use_similarity: whether to use cosine similarity for distractor selection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_size = len(train_features)
    valid_size = len(val_features) if val_features is not None else 0
    test_size = len(test_features) if test_features is not None else 0
    
    print(f"Generating {n_epochs} epoch datasets...")
    print(f"  Train: {train_size} samples")
    print(f"  Validation: {valid_size} samples")
    print(f"  Test: {test_size} samples")
    print(f"  Distractors: {n_distractors}")
    print(f"  Use similarity: {use_similarity}")
    
    for epoch in tqdm(range(n_epochs), desc="Generating epoch datasets"):
        print(f"\nEpoch {epoch}/{n_epochs-1}")
        
        # Create epoch-specific random state for reproducibility but different across epochs
        # Each epoch gets its own RNG seeded with seed + epoch
        epoch_rng = np.random.RandomState(seed + epoch)
        
        # Create training tuples (use all training samples)
        train_tuples, train_labels = create_curriculum_tuples_for_epoch(
            train_features,
            train_size,
            n_distractors,
            epoch,
            epoch_rng,
            use_similarity=use_similarity
        )
        
        # Create validation tuples (use all validation samples)
        if valid_size > 0:
            valid_tuples, valid_labels = create_curriculum_tuples_for_epoch(
                val_features,
                valid_size,
                n_distractors,
                epoch,
                epoch_rng,
                use_similarity=use_similarity
            )
        else:
            valid_tuples, valid_labels = np.array([]), np.array([])
        
        # Create test tuples (use all test samples)
        if test_size > 0:
            test_tuples, test_labels = create_curriculum_tuples_for_epoch(
                test_features,
                test_size,
                n_distractors,
                epoch,
                epoch_rng,
                use_similarity=use_similarity
            )
        else:
            test_tuples, test_labels = np.array([]), np.array([])
        
        # Save epoch data
        epoch_dir = output_path / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=True)
        
        epoch_file = epoch_dir / "data.npz"
        np.savez_compressed(
            epoch_file,
            train=train_tuples,
            train_labels=train_labels,
            valid=valid_tuples,
            valid_labels=valid_labels,
            test=test_tuples,
            test_labels=test_labels,
            epoch=epoch,
            n_distractors=n_distractors,
        )
        print(f"  Saved: {epoch_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate epoch-specific curriculum learning datasets from HuggingFace datasets"
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=False,
        help="Path to training dataset directory (loaded with datasets.load_from_disk)",
        default="/home/elena/emcomm/datasets/coco_train_features_resnet_152"
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        required=False,
        help="Path to validation dataset directory",
        default="/home/elena/emcomm/datasets/coco_val_features_resnet_152_splitted"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=False,
        help="Path to test dataset directory",
        default="/home/elena/emcomm/datasets/coco_test_features_resnet_152_splitted"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="epoch_datasets",
        help="Output directory for epoch datasets (default: epoch_datasets)"
    )
    parser.add_argument(
        "--feature_column",
        type=str,
        default="features",
        help="Name of the feature column in datasets (default: 'features')"
    )
    parser.add_argument(
        "--n_distractors",
        type=int,
        default=3,
        help="Number of distractors per tuple (default: 3)"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of epochs to generate (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no_similarity",
        action="store_true",
        help="Use random distractor selection instead of cosine similarity"
    )
    
    args = parser.parse_args()
    
    print("Loading datasets...")
    print(f"  Train dataset: {args.train_dataset_path}")
    train_features = load_dataset_features(args.train_dataset_path, args.feature_column)
    
    print(f"  Validation dataset: {args.val_dataset_path}")
    val_features = load_dataset_features(args.val_dataset_path, args.feature_column)
    
    print(f"  Test dataset: {args.test_dataset_path}")
    test_features = load_dataset_features(args.test_dataset_path, args.feature_column)
    
    generate_epoch_datasets(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        output_dir=args.output_dir,
        n_distractors=args.n_distractors,
        n_epochs=args.n_epochs,
        seed=args.seed,
        use_similarity=not args.no_similarity
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
