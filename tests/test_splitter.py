import numpy as np
import pytest
from models.splitter import purged_walk_forward_split


class TestPurgedWalkForwardSplit:
    """Tests for purged_walk_forward_split function."""

    @pytest.fixture
    def split_data(self):
        """Create test splits with standard parameters."""
        n = 1000
        n_folds = 5
        gap = 10
        splits = list(purged_walk_forward_split(n, n_folds=n_folds, gap=gap))
        return splits, n, n_folds, gap

    def test_fold_count(self, split_data):
        """Verify that the correct number of folds is generated."""
        splits, n, n_folds, gap = split_data
        assert len(splits) == n_folds

    def test_no_overlap(self, split_data):
        """Verify that train and val indices don't overlap within each fold."""
        splits, n, n_folds, gap = split_data
        for train_idx, val_idx in splits:
            overlap = np.intersect1d(train_idx, val_idx)
            assert len(overlap) == 0, "Train and validation sets should not overlap"

    def test_gap_enforced(self, split_data):
        """Verify that val_start > train_end (gap is actually enforced)."""
        splits, n, n_folds, gap = split_data
        for train_idx, val_idx in splits:
            train_end = train_idx[-1] if len(train_idx) > 0 else -1
            val_start = val_idx[0] if len(val_idx) > 0 else train_end
            # Ensure at least gap samples between train_end and val_start
            assert val_start > train_end, "val_start should be > train_end"
            assert val_start - train_end >= gap, f"Gap not enforced: {val_start} - {train_end} < {gap}"

    def test_expanding_window(self, split_data):
        """Verify that each fold's train set grows (expanding window)."""
        splits, n, n_folds, gap = split_data
        prev_train_size = 0
        for train_idx, val_idx in splits:
            current_train_size = len(train_idx)
            assert current_train_size >= prev_train_size, \
                "Train set should expand in subsequent folds"
            prev_train_size = current_train_size

    def test_val_covers_last_row(self, split_data):
        """Verify that the last fold's val includes the last index (n-1)."""
        splits, n, n_folds, gap = split_data
        last_train_idx, last_val_idx = splits[-1]
        assert n - 1 in last_val_idx, "Last fold should include the last index"

    def test_indices_are_numpy_arrays(self, split_data):
        """Verify that returned indices are numpy arrays."""
        splits, n, n_folds, gap = split_data
        for train_idx, val_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)

    def test_no_gaps_in_indices(self, split_data):
        """Verify that train and val indices are contiguous (no gaps within each set)."""
        splits, n, n_folds, gap = split_data
        for train_idx, val_idx in splits:
            # Train should be contiguous from 0
            if len(train_idx) > 0:
                assert np.array_equal(train_idx, np.arange(train_idx[0], train_idx[-1] + 1))
            # Val should be contiguous
            if len(val_idx) > 0:
                assert np.array_equal(val_idx, np.arange(val_idx[0], val_idx[-1] + 1))

    def test_specific_fold_boundaries(self):
        """Test specific fold boundaries with known values."""
        n = 1000
        n_folds = 5
        gap = 10
        fold_size = n // (n_folds + 1)  # fold_size = 166

        splits = list(purged_walk_forward_split(n, n_folds=n_folds, gap=gap))

        # Fold 1: train[0:166], gap, val[176:332]
        train_idx, val_idx = splits[0]
        assert train_idx[0] == 0
        assert train_idx[-1] == fold_size - 1
        assert val_idx[0] == fold_size + gap
        assert val_idx[-1] == fold_size * 2 - 1

        # Fold 5 (last): train[0:830], gap, val[840:1000]
        train_idx, val_idx = splits[4]
        assert train_idx[0] == 0
        assert train_idx[-1] == fold_size * 5 - 1
        assert val_idx[0] == fold_size * 5 + gap
        assert val_idx[-1] == n - 1
