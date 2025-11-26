from abc import ABC, abstractmethod

import numpy as np 
from typing import Any, List


class BaseSampler(ABC): 

    @abstractmethod
    def sample(self, items: List[Any], k: int, **kwargs: Any) -> List[Any]: 
        pass  

    def __call__(self, items: List[Any], k: int, **kwargs: Any) -> List[Any]: 
        return self.sample(items, k, **kwargs)

class UniformSampler(BaseSampler): 

    def sample(self, items: List[Any], k: int) -> List[Any]: 
        return np.random.choice(items, size=k, replace=False).tolist()
    
class WeightedSampler(BaseSampler): 

    def sample(self, items: List[Any], weights: List[float], k: int) -> List[Any]: 
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        return np.random.choice(items, size=k, replace=False, p=weights).tolist()
    
class SoftmaxSampler(BaseSampler): 

    def sample(self, items: List[Any], weights: List[float], k: int) -> List[Any]: 
        weights = np.array(weights, dtype=float)
        weights = np.exp(weights - weights.max())
        weights = weights / weights.sum()
        return np.random.choice(items, size=k, replace=False, p=weights).tolist()


class TopKSampler(BaseSampler):
    """
    Selects the top-k items with the highest weights.

    Args:
        items (List[Any]): List of item identifiers.
        weights (List[float]): List of weights corresponding to each item.
        k (int): Number of items to sample.

    Returns:
        List[Any]: The top-k items sorted by descending weight.
    """
    def sample(self, items: List[Any], weights: List[float], k: int) -> List[Any]:
        weights = np.array(weights, dtype=float)
        sorted_indices = np.argsort(weights)[::-1]
        topk_indices = sorted_indices[:k]
        return [items[i] for i in topk_indices]


class LastKSampler(BaseSampler):
    """
    Selects the last-k items with the lowest weights.

    Args:
        items (List[Any]): List of item identifiers.
        weights (List[float]): List of weights corresponding to each item.
        k (int): Number of items to sample.

    Returns:
        List[Any]: The last-k items sorted by ascending weight.
    """
    def sample(self, items: List[Any], weights: List[float], k: int) -> List[Any]:
        weights = np.array(weights, dtype=float)
        sorted_indices = np.argsort(weights)
        lastk_indices = sorted_indices[:k]
        return [items[i] for i in lastk_indices]


