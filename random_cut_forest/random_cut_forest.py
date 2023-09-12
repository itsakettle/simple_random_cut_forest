import numpy as np
from typing import NamedTuple, List
from random_cut_forest.binary_tree import BinaryTree

class RandomCutForest:
    
    data: np.ndarray
    tree: BinaryTree
    dim: int
    n: int

    def __init__(self, max_depth: int, data: np.ndarray):
        self.max_depth = max_depth
        self.n, self.dim = data.shape
        self.data = data
        self.tree: BinaryTree = BinaryTree(root_node_data=RandomCutForest.Cut(None, None, None))


    def fit(self):
        
        pass
    
    def make_cut(i:) -> :
        
    
    class Cut(NamedTuple):
        i: List[int]
        dim: int
        threshold: float