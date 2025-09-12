import os
import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data

class CaptionsFeat(data.Dataset):
    def __init__(self, root, train=True):
        # import h5py

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        # FC features
        fc_file = os.path.join(root, "ours_images_single_sm0.h5")

        fc = h5py.File(fc_file, "r")
        # There should be only 1 key
        key = list(fc.keys())[0]
        # Get the data
        data = torch.FloatTensor(list(fc[key]))

        # normalise data
        img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        normed_data = data / img_norm

        objects_file = os.path.join(root, "ours_images_single_sm0.objects")
        with open(objects_file, "rb") as f:
            labels = pickle.load(f)
        objects_file = os.path.join(root, "ours_images_paths_sm0.objects")
        with open(objects_file, "rb") as f:
            paths = pickle.load(f)

        self.create_obj2id(labels)
        self.data_tensor = normed_data
        self.labels = labels
        self.paths = paths

    def __getitem__(self, index):
        return self.data_tensor[index], index

    def __len__(self):
        return self.data_tensor.size(0)

    def create_obj2id(self, labels):
        self.obj2id = {}
        keys = {}
        idx_label = -1
        for i in range(labels.shape[0]):
            if not labels[i] in keys.keys():
                idx_label += 1
                keys[labels[i]] = idx_label
                self.obj2id[idx_label] = {}
                self.obj2id[idx_label]["labels"] = labels[i]
                self.obj2id[idx_label]["ims"] = []
            self.obj2id[idx_label]["ims"].append(i)