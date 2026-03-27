import numpy as np
from torch.utils import data
from src.objects_game.src.features import VectorsLoader, TupleDataset  


class MultiSplitVectorsLoader(VectorsLoader):
    def load_data_multi(self, data_file):
        data = np.load(data_file)

        # def maybe_get(name):
        #     return data[name].astype(np.float32) if name in data else None

        # def maybe_labels(name):
        #     return data[name] if name in data else None

        # train = maybe_get("train")
        # train_labels = maybe_labels("train_labels")

        # test = maybe_get("test")
        # test_labels = maybe_labels("test_labels")

        # collect validation splits
        valid_splits = {}

        for key, value in data.items():
            if key == "n_distractors" or key.endswith("_labels"):
                continue

            base = key
            labels_key = f"{base}_labels"

            # print(f"base: {base}")

            if labels_key in data:
                valid_splits[base] = (
                    value.astype(np.float32),
                    data[labels_key],
                )
        if not valid_splits:
            raise ValueError("No validation splits found")

        # infer metadata from any available split
        ref = list(valid_splits.values())[0][0]

        self.n_distractors = ref.shape[1] - 1
        self.perceptual_dimensions = [-1] * ref.shape[-1]
        self._n_features = ref.shape[-1]

        return valid_splits
        # return (
        #     (train, train_labels) if train is not None else None,
        #     valid_splits,
        #     (test, test_labels) if test is not None else None,
        # )

    def get_iterators(self):
        # train, valid_splits, test = self.load_data_multi(self.load_data_path)
        valid_splits = self.load_data_multi(self.load_data_path)

        def make_loader(split):
            if split is None:
                return None
            dataset = TupleDataset(*split)
            return data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                drop_last=True,
            )
        # train_it = make_loader(train)
        # test_it = make_loader(test)
        # validation_its = [make_loader(v) for v in valid_splits]
        validation_its = {
            name: make_loader(split)
            for name, split in valid_splits.items()
        }


        # return train_it, validation_its, test_it
        return validation_its