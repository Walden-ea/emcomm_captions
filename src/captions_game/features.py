import torch

import numpy as np



class CaptionsLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.opt = kwargs.pop("opt")
        self.seed = kwargs.pop("seed")
        self.batches_per_epoch = kwargs.pop("batches_per_epoch")

        super(CaptionsLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _BatchIterator(self, n_batches=self.batches_per_epoch, seed=seed)
    
class _BatchIterator:
    def __init__(self, loader, n_batches, seed=None):
        self.loader = loader
        self.n_batches = n_batches
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        loader = self.loader
        opt = loader.opt

        # images_indexes_sender = self.random_state.choice(
        #     len(loader.dataset), (opt.batch_size, opt.game_size), replace=False
        # )
        images_indexes_sender = np.stack([
        self.random_state.choice(len(loader.dataset), opt.game_size, replace=False)
        for _ in range(opt.batch_size)
        ])
        images_vectors_sender = []
        for i in range(opt.game_size):
            x = torch.tensor(loader.dataset['features'][images_indexes_sender[:, i]])
            images_vectors_sender.append(x)

        
        images_vectors_sender = torch.stack(images_vectors_sender).contiguous()
        y = torch.zeros(opt.batch_size).long()

        images_vectors_receiver = torch.zeros_like(images_vectors_sender)
        for i in range(opt.batch_size):
            permutation = torch.randperm(opt.game_size)

            images_vectors_receiver[:, i, :] = images_vectors_sender[permutation, i, :]
            y[i] = permutation.argmin()
        return images_vectors_sender.permute(1, 0, 2).contiguous(), y, images_vectors_receiver.permute(1, 0, 2).contiguous()

