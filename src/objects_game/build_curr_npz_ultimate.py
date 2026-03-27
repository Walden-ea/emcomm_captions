# %%
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
ds_wg = load_from_disk("/home/elena/emcomm/datasets/winoground_features_resnet_152")
# ds_wg = load_from_disk("/home/elena/emcomm/datasets/winoground_features_resnet_152")
ds_val = load_from_disk("/home/elena/emcomm/datasets/coco_val_features_resnet_152_splitted")
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
    i=0

    def build_tuple(target_vec, paired_vec, exclude_indices):
        # candidates excluding target & its pair
        nonlocal total_sim, total_paired_sim, count, i

        available_indices = np.setdiff1d(np.arange(num_total), exclude_indices)

        n_candidates = min(n_distractors + epoch, len(available_indices))
        if i==0:
            print(f"Num candidates: {n_candidates}")
            i+=1

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
# epoch = 0  # Set your curriculum learning epoch here
# num_epochs = 101
# print('doing train data now: ')
# for epoch in range(100, num_epochs):
#     # test_tuples, test_labels = create_exhaustive_tuples(np.array(ds_test["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
#     # valid_tuples, valid_labels = create_exhaustive_tuples(np.array(ds_val["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
#     # train_tuples, train_labels = create_exhaustive_tuples(np.array(ds_train["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
#     train_tuples, train_labels = create_pairwise_tuples(
#         ds_wg,
#         n_distractors=3,
#         epoch=epoch,
#         shuffle=True,
#         seed=42
#     )
#     valid_tuples, valid_labels = train_tuples, train_labels
#     test_tuples, test_labels = train_tuples, train_labels
#     np.savez_compressed(
#     f"/home/elena/emcomm/emcomm_captions/winoground_epochs/data_{n_distractors}_distractors_{epoch}_epoch.npz",
#     train=train_tuples,
#     train_labels=train_labels,
#     valid=valid_tuples,
#     valid_labels=valid_labels,
#     test=test_tuples,
#     test_labels=test_labels,
#     n_distractors=3
# )

def create_exhaustive_tuples(features, n_distractors=3, epoch=0, shuffle=True, seed=42):
    """
    Each object is used once as target.
    Distractors are selected based on semantic similarity with curriculum learning.
    
    The number of candidate distractors increases with epoch: n_distractors + epoch*2
    Then the n_distractors closest vectors by cosine similarity are selected.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        n_distractors: number of distractors per tuple
        epoch: curriculum learning epoch (affects difficulty)
        shuffle: whether to randomly shuffle tuple positions
        seed: random seed for reproducibility
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    num_objects = features.shape[0]
    rng = np.random.default_rng(seed + epoch)  # Different RNG per epoch

    tuples = []
    labels = []
    total_sim = 0.0
    count = 0

    all_indices = np.arange(num_objects)

    for i in tqdm(range(num_objects)):
        # Get all non-target indices
        non_target_indices = np.concatenate([
            np.arange(i),
            np.arange(i + 1, num_objects)
        ])
        
        # Number of candidates increases with epoch: n_distractors + epoch*2
        n_candidates = min(
            n_distractors + epoch,
            len(non_target_indices)
        )
        if i ==1:
            print(f"N candidates: {n_candidates}")
        
        # Sample candidate indices from non-target vectors
        # print(n_candidates)
        sampled_candidate_positions = rng.choice(
            np.arange(len(non_target_indices)),
            size=n_candidates,
            replace=False
        )
        sampled_candidate_indices = non_target_indices[sampled_candidate_positions]
        
        # Compute cosine similarity between target and sampled candidates
        target_vector = features[i:i+1]
        candidate_vectors = features[sampled_candidate_indices]
        similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
        
        # Select the n_distractors closest vectors by cosine similarity
        closest_positions = np.argsort(-similarities)[:n_distractors]
        distractor_idxs = sampled_candidate_indices[closest_positions]

        target = features[i]
        distractors = features[distractor_idxs]

        tuple_vectors = np.vstack([target[None, :], distractors])
        target_norm = target / np.linalg.norm(target)
        distractors_norm = distractors / np.linalg.norm(distractors, axis=1, keepdims=True)

        sims = distractors_norm @ target_norm
        total_sim += sims.mean()
        count += 1

        if shuffle:
            perm = rng.permutation(n_distractors + 1)
            tuple_vectors = tuple_vectors[perm]
            label = int(np.where(perm == 0)[0][0])
        else:
            label = 0

        tuples.append(tuple_vectors)
        labels.append(label)
    print(f"Average anchor–distractor similarity: {total_sim / count:.4f}")
    return np.array(tuples), np.array(labels)

wg_tuples, wg_labels = create_pairwise_tuples(
    ds_wg,
    n_distractors=n_distractors,
    epoch=9999999999999999,
    shuffle=True,
    seed=42
)

coc_easy_tuples, coco_easy_labels = create_exhaustive_tuples(
    np.array(ds_val["features"]),
    n_distractors=n_distractors,
    epoch=0,
    shuffle=True,
    seed=42,
)
coco_hard_tuples, coco_hard_labels = create_exhaustive_tuples(
    np.array(ds_val["features"]),
    n_distractors=n_distractors,
    epoch=99999999,
    shuffle=True,
    seed=42,
)

np.savez_compressed(
f"/home/elena/emcomm/emcomm_captions/combined_val/data_{n_distractors}_distractors_combined_val.npz",
    wg=wg_tuples,
    wg_labels=wg_labels,
    coco_easy=coc_easy_tuples,
    coco_easy_labels=coco_easy_labels,
    coco_hard=coco_hard_tuples,
    coco_hard_labels=coco_hard_labels,
    n_distractors=n_distractors
)