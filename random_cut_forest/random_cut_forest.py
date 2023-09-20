import numpy as np
from typing import Tuple, NamedTuple, List, Type
from random_cut_forest.binary_tree import BinaryTree

NodeDataType = Type["RandomCutForest.NodeData"]

class RandomCutForest:
    
    data: np.ndarray
    trees: List[BinaryTree]
    n_col: int
    n_row: int

    def __init__(self, data: np.ndarray, max_depth: int, min_node_size: int):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.n_row, self.n_col = data.shape
        self.data = data

    def _grow_a_tree(self, tree: BinaryTree = None) -> BinaryTree:
        
        if not tree:
          i = list(range(self.n_row))
          tree = BinaryTree(RandomCutForest.NodeData(i))
        nesting = "  "*tree.cursor.depth
        if (len(tree.cursor.data.i) <= self.min_node_size) or (tree.cursor.depth == self.max_depth):
            try:
                print(f"{nesting}move up")
                tree.move_up()
            except BinaryTree.Nodeless:
                pass
            
            return

        tree.cursor.data.col = self._choose_col()
        col_data = self.data[tree.cursor.data.i, tree.cursor.data.col]
        col_min = np.min(col_data)
        col_max = np.max(col_data)
        tree.cursor.data.threshold  = self._choose_threshold(col_min, col_max)
        left_i, right_i = self._make_cut(tree.cursor.data.col, 
                                        tree.cursor.data.threshold, 
                                        tree.cursor.data.i)
        tree.cursor.add_left_child(RandomCutForest.NodeData(left_i))
        tree.cursor.add_right_child(RandomCutForest.NodeData(right_i))
        tree.move_left()
        print(f"{nesting}go left")
        self._grow_a_tree(tree)
        print(f"{nesting}go right")
        tree.move_right()
        self._grow_a_tree(tree)

        return tree
        

    def fit(self):
        pass
    
    @classmethod
    def _choose_threshold(cls, min: float, max: float) -> float:
        return np.random.uniform(min, max)
    
    def _choose_col(self) -> int:
        return np.random.randint(0, self.n_col, size=1)[0]
    
    def _make_cut(self, col: int, threshold: float, i: List[int]) -> Tuple[List[int], List[int]]:
        data_to_cut = self.data[i][:, col]
        arr_i = np.array(i).reshape(-1, 1)
        data_to_cut_with_i = np.concatenate((data_to_cut.reshape(-1, 1), arr_i), axis=1)
        left_i = data_to_cut_with_i[data_to_cut <= threshold][:, 1].astype(int).tolist()
        right_i = data_to_cut_with_i[data_to_cut > threshold][:, 1].astype(int).tolist()
        return left_i, right_i

        
    
    class NodeData:
        i: List[int]
        col: int
        threshold: float

        def __init__(self, i):
            self.i = i