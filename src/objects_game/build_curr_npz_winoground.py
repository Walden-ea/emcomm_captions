# %%
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
ds_train = load_from_disk("/home/elena/emcomm/datasets/winoground_features_resnet_152")
# ds_val = load_from_disk("/home/elena/emcomm/datasets/coco_val_features_resnet_152_splitted")
# ds_test = load_from_disk("/home/elena/emcomm/datasets/coco_test_features_resnet_152_splitted")
# split = ds_val.train_test_split(test_size=1500, seed=42)
# ds_val, ds_test = split['test'], split['train']

# ds_train, ds_val, ds_test

# %%
# ds_test.save_to_disk("../../../datasets/coco_test_features_resnet_152_splitted")
# ds_val.save_to_disk("../../../datasets/coco_val_features_resnet_152_splitted")

# %%
def create_pairwise_tuples(ds, n_distractors=3, epoch=0, shuffle=True, seed=42):
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(seed + epoch)

    # Stack all features for global sampling
    features_0 = np.array(ds["features_0"])
    features_1 = np.array(ds["features_1"])

    all_features = np.vstack([features_0, features_1])
    num_total = all_features.shape[0]

    tuples, labels = [], []
    
    count = 0
    total_sim = 0.0
    total_paired_sim = 0.0

    def build_tuple(target_vec, paired_vec, exclude_indices):
        # candidates excluding target & its pair
        nonlocal total_sim, total_paired_sim, count

        available_indices = np.setdiff1d(np.arange(num_total), exclude_indices)

        n_candidates = min(n_distractors + epoch, len(available_indices))

        sampled = rng.choice(available_indices, size=n_candidates, replace=False)
        candidate_vectors = all_features[sampled]

        # similarity-based selection
        sims = cosine_similarity(target_vec[None, :], candidate_vectors).flatten()
        closest_idx = np.argsort(-sims)[:(n_distractors - 1)]

        distractors = candidate_vectors[closest_idx]

        # ALWAYS include paired vector
        distractors = np.vstack([paired_vec[None, :], distractors])

        tuple_vectors = np.vstack([target_vec[None, :], distractors])

        target_norm = target_vec / np.linalg.norm(target_vec)
        distractors_norm = distractors / np.linalg.norm(distractors, axis=1, keepdims=True)

        sims = distractors_norm @ target_norm  # shape: (n_distractors,)
        total_sim += sims.mean()
        paired_sim = sims[0]  # because you stacked it first
        total_paired_sim += paired_sim

        count += 1

        if shuffle:
            perm = rng.permutation(n_distractors + 1)
            tuple_vectors = tuple_vectors[perm]
            label = int(np.where(perm == 0)[0][0])
        else:
            label = 0

        return tuple_vectors, label

    for i in tqdm(range(len(ds))):
        f0 = features_0[i]
        f1 = features_1[i]

        # indices in stacked array
        idx_0 = i
        idx_1 = i + len(ds)

        # case 1: target = f0
        t, l = build_tuple(
            target_vec=f0,
            paired_vec=f1,
            exclude_indices=[idx_0, idx_1]
        )
        tuples.append(t)
        labels.append(l)

        # case 2: target = f1
        t, l = build_tuple(
            target_vec=f1,
            paired_vec=f0,
            exclude_indices=[idx_0, idx_1]
        )
        tuples.append(t)
        labels.append(l)
    print(f"Average anchor–distractor similarity: {total_sim / count:.4f}")
    print(f"Avg paired sim:     {total_paired_sim / count:.4f}")
    return np.array(tuples), np.array(labels)

# %%
n_distractors = 3
epoch = 0  # Set your curriculum learning epoch here
num_epochs = 101
print('doing train data now: ')
for epoch in range(100, num_epochs):
    # test_tuples, test_labels = create_exhaustive_tuples(np.array(ds_test["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    # valid_tuples, valid_labels = create_exhaustive_tuples(np.array(ds_val["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    # train_tuples, train_labels = create_exhaustive_tuples(np.array(ds_train["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    train_tuples, train_labels = create_pairwise_tuples(
        ds_train,
        n_distractors=3,
        epoch=epoch,
        shuffle=True,
        seed=42
    )
    valid_tuples, valid_labels = train_tuples, train_labels
    test_tuples, test_labels = train_tuples, train_labels
    np.savez_compressed(
    f"/home/elena/emcomm/emcomm_captions/winoground_epochs/data_{n_distractors}_distractors_{epoch}_epoch.npz",
    train=train_tuples,
    train_labels=train_labels,
    valid=valid_tuples,
    valid_labels=valid_labels,
    test=test_tuples,
    test_labels=test_labels,
    n_distractors=3
)




