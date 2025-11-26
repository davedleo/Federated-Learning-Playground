from collections import defaultdict 

import numpy as np 
from typing import Any, Dict, List, Tuple

class Dirichlet:
    """
    Splits training and test data across multiple splits according to a Dirichlet
    distribution fitted on the class proportions of the training data.

    Parameters
    ----------
    train_data : List[Tuple[Any, int]]
        A list of (sample, label) pairs for training.
    test_data : List[Tuple[Any, int]]
        A list of (sample, label) pairs for testing.
    alpha : float, optional
        Dirichlet concentration parameter. Smaller → more skewed class allocation.
    seed : int, optional
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        train_data: List[Tuple[Any, int]],
        test_data: List[Tuple[Any, int]],
        alpha: float = 1.0
    ):
        self.alpha = alpha

        # Group training samples by label
        self.train_data = defaultdict(list)
        for x, y in train_data:
            self.train_data[y].append((x, y))

        # Group test samples by label
        self.test_data = defaultdict(list)
        for x, y in test_data:
            self.test_data[y].append((x, y))

    def _sample_class_splits(self, label_data: Dict[int, List[Tuple[Any, int]]], n_splits: int):
        """Return a dictionary mapping client → allocated samples for given label_data."""
        client_alloc = {i: [] for i in range(n_splits)}
        labels = list(label_data.keys())

        # Draw class-level proportions for each client
        proportions = np.random.dirichlet([self.alpha] * n_splits, size=len(labels))

        # For each class, allocate samples to splits
        for class_idx, c in enumerate(labels):
            samples = label_data[c]
            count = len(samples)

            # Shuffle samples
            shuffled = samples.copy()
            np.random.shuffle(shuffled)

            # Compute split boundaries
            class_props = proportions[class_idx]
            cum_props = np.cumsum(class_props)
            split_indices = (cum_props * count).astype(int)

            # Convert splits to lists of tuples
            splits = [[tuple(item) for item in chunk] for chunk in np.split(np.array(shuffled, dtype=object), split_indices[:-1])]

            for client_id, chunk in enumerate(splits):
                client_alloc[client_id].extend(chunk)

        return client_alloc

    def _avoid_empty_splits(self, alloc: Dict[int, List[Tuple[Any, int]]], label_data: Dict[int, List[Tuple[Any, int]]]):
        """Ensure no client is empty. Borrow samples from the client with the most samples until all clients have at least one."""
        empty_clients = [cid for cid, samples in alloc.items() if len(samples) == 0]

        while empty_clients:
            for client_id in empty_clients:
                # Find donor client with maximum items
                donor = max(alloc, key=lambda k: len(alloc[k]))
                if len(alloc[donor]) > 1:
                    alloc[client_id].append(alloc[donor].pop())
                else:
                    # If donor can't give, pick any non-empty client
                    for k in alloc:
                        if len(alloc[k]) > 0:
                            alloc[client_id].append(alloc[k].pop())
                            break
            empty_clients = [cid for cid, samples in alloc.items() if len(samples) == 0]

        return alloc

    def sample(self, n_splits: int) -> List[Dict[str, List[Tuple[Any, int]]]]:
        """
        Produce non-IID train-test splits for a given number of splits.
        Ensures no client is empty.

        Output format:
        [
            {"train": [...], "test": [...]},
            {"train": [...], "test": [...]},
            ...
        ]
        """

        # Sample from training data
        train_alloc = self._sample_class_splits(self.train_data, n_splits)
        train_alloc = self._avoid_empty_splits(train_alloc, self.train_data)

        # Sample test splits using SAME fitted Dirichlet proportions:
        # → The same proportions used for training should define test splits.
        # Extract final proportions = sizes of train splits by class
        labels = list(self.train_data.keys())

        # Compute empirical client-class proportions from the generated train split
        empirical_props = {c: [] for c in labels}
        for c in labels:
            total = len(self.train_data[c])
            if total == 0:
                empirical_props[c] = np.ones(n_splits) / n_splits
                continue
            counts = np.array([sum(1 for x, y in train_alloc[i] if y == c) for i in range(n_splits)])
            if counts.sum() == 0:
                empirical_props[c] = np.ones(n_splits) / n_splits
            else:
                empirical_props[c] = counts / counts.sum()

        # Use empirical_props to split test data deterministically
        test_alloc = {i: [] for i in range(n_splits)}
        for c in labels:
            samples = self.test_data[c]
            shuffled = samples.copy()
            np.random.shuffle(shuffled)

            count = len(samples)
            proportions = empirical_props[c]
            cum_props = np.cumsum(proportions)
            split_indices = (cum_props * count).astype(int)
            splits = [[tuple(item) for item in chunk] for chunk in np.split(np.array(shuffled, dtype=object), split_indices[:-1])]
            for client_id, chunk in enumerate(splits):
                test_alloc[client_id].extend(chunk)

        test_alloc = self._avoid_empty_splits(test_alloc, self.test_data)

        # Final format
        return [
            {"train": train_alloc[i], "test": test_alloc[i]}
            for i in range(n_splits)
        ]