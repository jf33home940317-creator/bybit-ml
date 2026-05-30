from typing import Iterator
import numpy as np


def purged_walk_forward_split(
    n: int,
    n_folds: int = 5,
    gap: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Purged Walk-Forward Cross-Validation Split.

    Generates expanding window train-validation splits with a gap between them
    to prevent look-ahead bias in time series modeling.

    Args:
        n: Total number of samples
        n_folds: Number of folds to generate
        gap: Number of samples to skip between train and validation sets

    Yields:
        Tuples of (train_indices, val_indices) as numpy arrays

    Example:
        >>> for train_idx, val_idx in purged_walk_forward_split(1000, n_folds=5, gap=10):
        ...     print(len(train_idx), len(val_idx))
    """
    fold_size = n // (n_folds + 1)

    for fold_idx in range(n_folds):
        # Training set: from start to fold_size * (fold_idx + 1)
        train_end = fold_size * (fold_idx + 1)
        train_indices = np.arange(0, train_end)

        # Validation set: from train_end + gap to fold_size * (fold_idx + 2)
        # For the last fold, extend to the end of the data
        val_start = train_end + gap
        if fold_idx < n_folds - 1:
            val_end = fold_size * (fold_idx + 2)
        else:
            val_end = n

        val_indices = np.arange(val_start, val_end)

        yield train_indices, val_indices
