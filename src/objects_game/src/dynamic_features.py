import numpy as np
import torch
from torch.utils import data


class DynamicTupleDataset(data.Dataset):
    """
    Dataset that generates tuples dynamically on the fly with random distractors.
    Each epoch will have different random selections of distractors for the same targets.
    
    Args:
        features: numpy array of shape (num_objects, feature_dim)
        n_distractors: number of distractor objects per tuple
        n_samples: number of tuples to generate (approximate, actual number depends on seed)
        shuffle: whether to shuffle the target position in the tuple
        seed: random seed for reproducibility (can vary per epoch)
    """
    
    def __init__(self, features, n_distractors=3, n_samples=None, shuffle=True, seed=None):
        self.features = np.array(features)
        self.n_distractors = n_distractors
        self.shuffle = shuffle
        self.num_objects = len(self.features)
        
        # If n_samples not specified, use all objects as targets
        self.n_samples = n_samples if n_samples is not None else self.num_objects
        
        # Set seed for reproducibility
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def set_epoch(self, epoch):
        """
        Set seed based on epoch to get different distractors each epoch.
        Call this at the start of each epoch if you want epoch-specific randomization.
        """
        self.seed = epoch
        self.rng = np.random.default_rng(epoch)
    
    def _generate_tuple(self, target_idx):
        """
        Generate a single tuple with a given target and random distractors.
        
        Returns:
            tuple_vectors: numpy array of shape (n_distractors+1, feature_dim)
            label: index of the target in the shuffled tuple (0 to n_distractors)
        """
        all_indices = np.arange(self.num_objects)
        candidates = np.delete(all_indices, target_idx)
        distractor_idxs = self.rng.choice(candidates, size=self.n_distractors, replace=False)
        
        target = self.features[target_idx]
        distractors = self.features[distractor_idxs]
        
        # Stack target with distractors
        tuple_vectors = np.vstack([target[None, :], distractors])
        
        # Optionally shuffle and track position
        if self.shuffle:
            perm = self.rng.permutation(self.n_distractors + 1)
            tuple_vectors = tuple_vectors[perm]
            label = int(np.where(perm == 0)[0][0])
        else:
            label = 0
        
        return tuple_vectors, label
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Map index to a target object
        # If n_samples > num_objects, we'll cycle through and potentially repeat targets
        target_idx = idx % self.num_objects
        
        tuple_vectors, label = self._generate_tuple(target_idx)
        return tuple_vectors, label


class DynamicVectorsLoader:
    """
    Data loader that generates tuples dynamically on each epoch.
    Works with raw features rather than pre-generated tuples.
    
    Args:
        train_features: numpy array of training features
        valid_features: numpy array of validation features
        test_features: numpy array of test features
        n_distractors: number of distractors per tuple
        batch_size: batch size for DataLoader
        train_samples: number of training tuples to generate
        validation_samples: number of validation tuples to generate
        test_samples: number of test tuples to generate
        shuffle_train_data: whether to shuffle training data
        seed: random seed
    """
    
    def __init__(
        self,
        train_features,
        valid_features=None,
        test_features=None,
        n_distractors=3,
        batch_size=32,
        train_samples=None,
        validation_samples=None,
        test_samples=None,
        shuffle_train_data=False,
        seed=None,
    ):
        self.train_features = np.array(train_features)
        self.valid_features = np.array(valid_features) if valid_features is not None else None
        self.test_features = np.array(test_features) if test_features is not None else None
        
        self.n_distractors = n_distractors
        self.batch_size = batch_size
        self.shuffle_train_data = shuffle_train_data
        
        # Set sample counts
        self.train_samples = train_samples if train_samples is not None else len(self.train_features)
        self.validation_samples = validation_samples if validation_samples is not None else (
            len(self.valid_features) if self.valid_features is not None else 0
        )
        self.test_samples = test_samples if test_samples is not None else (
            len(self.test_features) if self.test_features is not None else 0
        )
        
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self.seed = seed
    
    def collate(self, batch):
        """
        Collate function for DataLoader.
        Converts batch of (tuple_vectors, label) into batched tensors.
        """
        tuples, target_idxs = [elem[0] for elem in batch], [elem[1] for elem in batch]
        receiver_input = np.reshape(
            tuples, (self.batch_size, self.n_distractors + 1, -1)
        )
        labels = np.array(target_idxs)
        targets = receiver_input[np.arange(self.batch_size), labels]
        return (
            torch.from_numpy(targets).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(receiver_input).float(),
        )
    
    def get_iterators(self, epoch=None):
        """
        Create DataLoaders for train, validation, and test sets.
        
        Args:
            epoch: optional epoch number for deterministic per-epoch randomization
        
        Returns:
            tuple of (train_iterator, validation_iterator, test_iterator)
        """
        # Use epoch as seed offset if provided
        seed_offset = 0 if epoch is None else epoch
        
        # Create datasets
        train_dataset = DynamicTupleDataset(
            self.train_features,
            n_distractors=self.n_distractors,
            n_samples=self.train_samples,
            shuffle=True,
            seed=self.seed + seed_offset,
        )
        
        train_it = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
            shuffle=self.shuffle_train_data,
        )
        
        validation_it = None
        if self.valid_features is not None:
            valid_dataset = DynamicTupleDataset(
                self.valid_features,
                n_distractors=self.n_distractors,
                n_samples=self.validation_samples,
                shuffle=True,
                seed=self.seed + 1000000 + seed_offset,
            )
            validation_it = data.DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                drop_last=True,
            )
        
        test_it = None
        if self.test_features is not None:
            test_dataset = DynamicTupleDataset(
                self.test_features,
                n_distractors=self.n_distractors,
                n_samples=self.test_samples,
                shuffle=True,
                seed=self.seed + 2000000 + seed_offset,
            )
            test_it = data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                drop_last=True,
            )
        
        return train_it, validation_it, test_it
