import pytest
from random_cut_forest.binary_tree import BinaryTree

@pytest.fixture
def data():
    return [list(range(100)), list(range(200))]

def test_create_new_tree(data):
    tree = BinaryTree(root_branch_data=data[0])
    assert tree.root_node.parent is None
    assert len(tree.root_node.data) == 100

def test_add_branches_to_tree_and_move(data):
    tree = BinaryTree(root_branch_data=data[0])
    tree.cursor.add_left_child(data[1])
    tree.move_left()
    assert len(tree.cursor.data) == 200
    tree.move_up()
    assert len(tree.cursor.data) == 100

def test_move_errors(data):
    tree = BinaryTree(root_branch_data=data[0])
    tree.cursor.add_left_child(data[1])
    
    with pytest.raises(BinaryTree.Nodeless):
        tree.move_right()

    with pytest.raises(BinaryTree.Nodeless):
        tree.move_up()


def test_add_branches_to_tree(data):
    tree = BinaryTree(root_branch_data=data[0])
    tree.cursor.add_left_child(data[1])
    tree.move_left()
    assert len(tree.cursor.data) == 200

def test_add_branch_to_tree(data):
    branch = BinaryTree.Node(data=data[0], parent=None)
    assert branch.is_leaf == True
    assert branch.child_count == 0
    branch.add_left_child(data[1])
    assert branch.is_leaf == False
    assert branch.child_count == 1
    branch.add_right_child(data[0])
    assert branch.is_leaf == False
    assert branch.child_count == 2
    
    
    
