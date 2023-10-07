import pytest
import numpy as np
from random_cut_forest.random_cut_forest import RandomCutForest
from itertools import cycle

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
    rcf = RandomCutForest(data=data_3_dim, max_depth=10, min_node_size=1, ntree=2)
    cuts = rcf._make_cut(2, 4.2, list(range(rcf.n_row)))
    assert cuts[0] == [0, 3]
    assert cuts[1] == [1, 2, 4, 5, 6, 7, 8, 9, 10]

@pytest.fixture
def mock_threshold(mocker):
    mock = mocker.patch.object(RandomCutForest, '_choose_threshold')
    mock.return_value = 4.3
    yield mock

@pytest.fixture
def mock_col(mocker):
    def mocked_choose_col(self):
        col_cycle = cycle([0, 1, 2])
        while True:
            yield next(col_cycle)
    
    mock = mocker.patch.object(RandomCutForest, '_choose_col_generator', mocked_choose_col)
    yield mock

def test_grow_one_tree(data_3_dim, mock_threshold, mock_col):
    rcf = RandomCutForest(data=data_3_dim, max_depth=2, min_node_size=2, ntree=10)
    tree = rcf._grow_a_tree()

    assert len(tree.cursor.left_child.data.i) == 3
    assert len(tree.cursor.right_child.data.i) == 8
    
    tree.move_left()
    assert len(tree.cursor.left_child.data.i) == 3
    assert len(tree.cursor.right_child.data.i) == 0
    assert tree.cursor.right_child.is_leaf == True
    
    tree.move_left()
    assert len(tree.cursor.left_child.data.i) == 2
    assert len(tree.cursor.right_child.data.i) == 1
    assert tree.cursor.left_child.is_leaf == True
    assert tree.cursor.right_child.is_leaf == True

def test_fit_forest(data_3_dim):
    rcf = RandomCutForest(data=data_3_dim, max_depth=2, min_node_size=2, ntree=100)
    rcf.fit()
    
    assert len(rcf.trees) == 100

    print(rcf.scores)
