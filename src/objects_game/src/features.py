# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import pathlib
from functools import reduce

import numpy as np
import torch
from torch.utils import data

from egg.zoo.objects_game.util import compute_binomial


class VectorsLoader:
    def __init__(
        self,
        perceptual_dimensions=[4, 4, 4, 4, 4],
        n_distractors=1,
        batch_size=32,
        train_samples=128000,
        validation_samples=4096,
        test_samples=1024,
        shuffle_train_data=False,
        dump_data_folder=None,
        load_data_path=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        seed=None,
    ):

        self.perceptual_dimensions = perceptual_dimensions
        self._n_features = (
            len(self.perceptual_dimensions) if self.perceptual_dimensions else None
        )
        self.n_distractors = n_distractors

        self.batch_size = batch_size
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.test_samples = test_samples

        self.shuffle_train_data = shuffle_train_data

        self.load_data_path = load_data_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.dump_data_folder = (
            pathlib.Path(dump_data_folder) if dump_data_folder is not None else None
        )

        seed = seed if seed else np.random.randint(0, 2**31)
        self.random_state = np.random.RandomState(seed)

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    def upd_cl_options(self, opts):
        opts.perceptual_dimensions = self.perceptual_dimensions
        opts.train_samples = self.train_samples
        opts.validation_samples = self.validation_samples
        opts.test_samples = self.test_samples
        opts.n_distractors = self.n_distractors

    def load_data(self, data_file):
        data = np.load(data_file)
        train, train_labels = data["train"], data["train_labels"]
        valid, valid_labels = data["valid"], data["valid_labels"]
        test, test_labels = data["test"], data["test_labels"]

        # train valid and test are of shape b_size X n_distractors+1 X n_features
        self.train_samples = train.shape[0]
        self.validation_samples = valid.shape[0]
        self.test_samples = test.shape[0]

        self.n_distractors = train.shape[1] - 1
        self.perceptual_dimensions = [-1] * train.shape[-1]
        self._n_features = len(self.perceptual_dimensions)

        return (train, train_labels), (valid, valid_labels), (test, test_labels)

    def _extract_features_from_dataset(self, dataset):
        """Extract features from a single dataset and convert to numpy array.

        Args:
            dataset: Dataset object with 'features' column

        Returns:
            numpy array of feature vectors
        """
        if dataset is None:
            return np.array([])

        # Extract features and convert to consistent format
        all_features = []
        if isinstance(dataset["features"][0], (list, tuple)):
            all_features = [np.array(f) for f in dataset["features"]]
        else:
            all_features = [dataset["features"][i] for i in range(len(dataset))]

        all_vectors = np.array(all_features)

        # Update n_features based on the actual feature dimension
        self._n_features = all_vectors.shape[1] if len(all_vectors.shape) > 1 else 1

        return all_vectors

    def _fill_split(self, all_vectors, n_samples, tuple_dict):
        split_list = []
        len_all_vectors = len(all_vectors)
        tuple_dim = self.n_distractors + 1
        done = 0
        while done < n_samples:
            candidates_tuple = self.random_state.choice(
                len_all_vectors, replace=False, size=tuple_dim
            )
            key = ""
            for vector_idx in candidates_tuple:
                key += f"{str(vector_idx)}-"
            key = key[:-1]
            if key not in tuple_dict:
                tuple_dict[key] = True
                possible_batch = all_vectors[candidates_tuple]
                split_list.append(possible_batch)
                done += 1
            else:
                continue

        target_idxs = self.random_state.choice(self.n_distractors + 1, n_samples)

        return (np.array(split_list), target_idxs), tuple_dict

    def _fill_split_from_dataset(self, all_vectors):
        """Create tuples ensuring each sample serves as a target exactly once.
        
        Creates as many datapoints as there are samples in all_vectors.
        Each sample will serve as the target vector exactly once.
        
        Args:
            all_vectors: numpy array of shape (n_samples, n_features)
            
        Returns:
            tuple of (split_data, target_idxs)
            - split_data: numpy array of shape (n_samples, n_distractors+1, n_features)
            - target_idxs: numpy array of shape (n_samples,) with values 0 to n_distractors
        """
        n_samples = len(all_vectors)
        tuple_dim = self.n_distractors + 1
        
        # Pre-allocate output array (much faster than appending)
        split_list = np.zeros((n_samples, tuple_dim, all_vectors.shape[1]), dtype=all_vectors.dtype)
        
        # Assign random target positions for each sample
        target_idxs = self.random_state.randint(0, tuple_dim, n_samples)
        
        # Place all targets at once using advanced indexing
        split_list[np.arange(n_samples), target_idxs] = all_vectors
        
        # Fill distractor positions
        for target_idx in range(n_samples):
            target_pos = target_idxs[target_idx]
            
            # Get non-target indices efficiently
            non_target_indices = np.concatenate([
                np.arange(target_idx),
                np.arange(target_idx + 1, n_samples)
            ])
            
            # Sample distractors
            distractor_indices = self.random_state.choice(
                non_target_indices,
                size=self.n_distractors,
                replace=False
            )
            
            # Get positions to fill (all except target position)
            fill_positions = [i for i in range(tuple_dim) if i != target_pos]
            
            # Place all distractors at once
            split_list[target_idx, fill_positions] = all_vectors[distractor_indices]
        
        return (split_list, target_idxs)

    def generate_tuples(self, data):
        data = np.array(data)
        train_data, tuple_dict = self._fill_split(data, self.train_samples, {})
        valid_data, tuple_dict = self._fill_split(
            data, self.validation_samples, tuple_dict
        )
        test_data, _ = self._fill_split(data, self.test_samples, tuple_dict)
        return train_data, valid_data, test_data

    def collate(self, batch):
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

    def get_iterators(self):
        if (
            self.train_dataset is not None
            or self.val_dataset is not None
            or self.test_dataset is not None
        ):
            # Extract features from each dataset separately and generate tuples for each

            # Process training dataset
            if self.train_dataset is not None:
                train_vectors = self._extract_features_from_dataset(self.train_dataset)
                train = self._fill_split_from_dataset(train_vectors)
                self.train_samples = len(train_vectors)
            else:
                train = (np.array([]), np.array([]))

            # Process validation dataset
            if self.val_dataset is not None:
                val_vectors = self._extract_features_from_dataset(self.val_dataset)
                valid = self._fill_split_from_dataset(val_vectors)
                self.validation_samples = len(val_vectors)
            else:
                valid = (np.array([]), np.array([]))

            # Process test dataset
            if self.test_dataset is not None:
                test_vectors = self._extract_features_from_dataset(self.test_dataset)
                test = self._fill_split_from_dataset(test_vectors)
                self.test_samples = len(test_vectors)
            else:
                test = (np.array([]), np.array([]))
        elif self.load_data_path:
            train, valid, test = self.load_data(self.load_data_path)
        else:
            raise ValueError(
                "Either train_dataset, val_dataset, test_dataset must be provided, "
                "or load_data_path must be specified"
            )

        assert (
            self.train_samples >= self.batch_size
            and self.validation_samples >= self.batch_size
            and self.test_samples >= self.batch_size
        ), f"\
Batch size cannot be smaller than any split size;\nGot batch size {self.batch_size}, \ntrain samples {self.train_samples},\n\nvalidation samples {self.validation_samples},\ntest samples {self.test_samples}\
        "

        train_dataset = TupleDataset(*train)
        valid_dataset = TupleDataset(*valid)
        test_dataset = TupleDataset(*test)

        train_it = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
            shuffle=self.shuffle_train_data,
        )
        validation_it = data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
        )
        test_it = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
        )

        if self.dump_data_folder:
            self.dump_data_folder.mkdir(exist_ok=True)
            path = (
                self.dump_data_folder
                / f"{self.perceptual_dimensions}_{self.n_distractors}_distractors"
            )
            np.savez_compressed(
                path,
                train=train[0],
                train_labels=train[1],
                valid=valid[0],
                valid_labels=valid[1],
                test=test[0],
                test_labels=test[1],
                n_distractors=self.n_distractors,
            )

        return train_it, validation_it, test_it


class TupleDataset(data.Dataset):
    def __init__(self, tuples, target_idxs):
        self.list_of_tuples = tuples
        self.target_idxs = target_idxs

    def __len__(self):
        return len(self.list_of_tuples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.list_of_tuples):
            raise RuntimeError(
                "Accessing dataset through wrong index: < 0 or >= max_len"
            )
        return self.list_of_tuples[idx], self.target_idxs[idx]
