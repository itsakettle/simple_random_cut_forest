import pytest
import numpy as np
from random_cut_forest.random_cut_forest import RandomCutForest

@pytest.fixture
def data_3_dim():
    data = [[1.0, 3.0, 2.0], 
            [33.0, 1.0, 11.0],
            [3.0, 1.0, 12.0],
            [4.0, 2.0, -13.0],
            [5.0, 3.0, 11.0],
            [6.0, 2.0, 9.0],
            [7.0, 3.0, 10.0],
            [8.0, 2.0, 12.0],
            [9.0, 3.0, 13.0],
            [8.0, 2.0, 11.0],
            [7.0, 3.0, 12.0],]

    return np.array(data)

def test_make_cut(data_3_dim):
    rcf = RandomCutForest(data=data_3_dim, max_depth=10, min_node_size=1)
    cuts = rcf._make_cut(2, 4.2, list(range(rcf.n_row)))
    assert cuts[0] == [0, 3]
    assert cuts[1] == [1, 2, 4, 5, 6, 7, 8, 9, 10]

def test_grow_one_tree(data_3_dim):
    rcf = RandomCutForest(data=data_3_dim, max_depth=10, min_node_size=1)
    tree = rcf._grow_a_tree()
    print(tree.cursor.data.threshold)

    
