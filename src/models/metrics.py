from abc import ABC, abstractmethod

import numpy as np
from typing import List


class BaseMetric(ABC): 

    def __init__(
            self, 
            metric_id: str
    ): 
        self._id = metric_id

    @property 
    def id(self): 
        return self._id
    
    def __call__(
            self,
            targets: List[int],
            preds: List[List[int]]
    ) -> float: 
        return self.forward(targets, preds)
    
    @abstractmethod 
    def forward(
            self,
            targets: List[int],
            preds: List[List[int]]
    ) -> float: 
        pass

class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__("accuracy")

    def forward(self, preds, targets):
        targets = np.array(targets, dtype=int)
        preds = np.array(preds, dtype=float).argmax(1)
        return float((targets == preds).mean())