# %%
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
ds_train = load_from_disk("/home/elena/emcomm/datasets/coco_train_features_resnet_152")
ds_val = load_from_disk("/home/elena/emcomm/datasets/coco_val_features_resnet_152_splitted")
ds_test = load_from_disk("/home/elena/emcomm/datasets/coco_test_features_resnet_152_splitted")
# split = ds_val.train_test_split(test_size=1500, seed=42)
# ds_val, ds_test = split['test'], split['train']

ds_train, ds_val, ds_test

# %%
# ds_test.save_to_disk("../../../datasets/coco_test_features_resnet_152_splitted")
# ds_val.save_to_disk("../../../datasets/coco_val_features_resnet_152_splitted")

# %%
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

        if shuffle:
            perm = rng.permutation(n_distractors + 1)
            tuple_vectors = tuple_vectors[perm]
            label = int(np.where(perm == 0)[0][0])
        else:
            label = 0

        tuples.append(tuple_vectors)
        labels.append(label)

    return np.array(tuples), np.array(labels)

# %%
n_distractors = 3
epoch = 0  # Set your curriculum learning epoch here
num_epochs = 10001
print('doing train data now: ')
# curr= [150, 300, 750, 1000, 1500, 2000, 3000, 7500, 12000]
curr =[1500]
for epoch in curr:
    test_tuples, test_labels = create_exhaustive_tuples(np.array(ds_test["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    valid_tuples, valid_labels = create_exhaustive_tuples(np.array(ds_val["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    train_tuples, train_labels = test_tuples, test_labels
    # train_tuples, train_labels = create_exhaustive_tuples(np.array(ds_train["features"]), n_distractors=n_distractors, epoch=epoch, shuffle=True, seed=42)
    np.savez_compressed(
    f"/home/elena/emcomm/emcomm_captions/epoch_datasets_slower/hard_test_data_dummy_train_{n_distractors}_distractors_{epoch}_epoch.npz",
    train=train_tuples,
    train_labels=train_labels,
    valid=valid_tuples,
    valid_labels=valid_labels,
    test=test_tuples,
    test_labels=test_labels,
    n_distractors=3
)




