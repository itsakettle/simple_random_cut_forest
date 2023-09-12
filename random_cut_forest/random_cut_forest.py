import numpy as np
from typing import int

class RandomCutForest:
    
    data: np.ndarray

    def __init__(self, max_depth, min_elements):
        self.max_depth = max_depth
        self.min_elements = min_elements

    def fit(self, data):
        pass