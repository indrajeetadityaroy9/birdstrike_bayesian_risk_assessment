"""Stratified train/val/test splitting ensuring species representation.

Each species is split proportionally to maintain distribution across splits.
"""


import numpy as np


def stratified_split(
    species: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/val/test splits ensuring species representation.

    Each species is split proportionally to maintain distribution across splits.

    Args:
        species: [N] species array
        val_split: Fraction for validation
        test_split: Fraction for test
        seed: Random seed

    Returns:
        (train_idx, val_idx, test_idx) index arrays

    """
    rng = np.random.RandomState(seed)

    # Collect arrays per species, then concatenate (avoids repeated list reallocation)
    train_arrays, val_arrays, test_arrays = [], [], []

    unique_species = np.unique(species)
    for sp in unique_species:
        sp_indices = np.where(species == sp)[0]
        rng.shuffle(sp_indices)

        n_sp = len(sp_indices)
        n_test = max(1, int(n_sp * test_split))  # At least 1 sample
        n_val = max(1, int(n_sp * val_split))
        n_train = n_sp - n_test - n_val

        # Ensure at least 1 training sample
        if n_train < 1:
            n_train = 1
            n_val = max(0, n_sp - n_train - n_test)
            if n_val < 0:
                n_test = n_sp - n_train
                n_val = 0

        if n_train > 0:
            train_arrays.append(sp_indices[:n_train])
        if n_val > 0:
            val_arrays.append(sp_indices[n_train:n_train + n_val])
        if n_sp - n_train - n_val > 0:
            test_arrays.append(sp_indices[n_train + n_val:])

    # Concatenate all species arrays at once (single allocation per split)
    train_idx = np.concatenate(train_arrays) if train_arrays else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_arrays) if val_arrays else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_arrays) if test_arrays else np.array([], dtype=np.int64)

    # Shuffle within each split to avoid species ordering artifacts
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx
