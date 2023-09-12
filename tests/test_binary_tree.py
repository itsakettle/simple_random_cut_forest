import pytest
from random_cut_forest.binary_tree import BinaryTree

@pytest.fixture
def data():
    return [list(range(100)), list(range(200))]

def test_create_new_tree(data):
    tree = BinaryTree(root_node_data=data[0])
    assert tree.root_node.parent is None
    assert len(tree.root_node.data) == 100

def test_add_branches_to_tree_and_move(data):
    tree = BinaryTree(root_node_data=data[0])
    tree.cursor.add_left_child(data[1])
    tree.move_left()
    assert len(tree.cursor.data) == 200
    tree.move_up()
    assert len(tree.cursor.data) == 100

def test_move_errors(data):
    tree = BinaryTree(root_node_data=data[0])
    tree.cursor.add_left_child(data[1])
    
    with pytest.raises(BinaryTree.Nodeless):
        tree.move_right()

    with pytest.raises(BinaryTree.Nodeless):
        tree.move_up()


def test_add_branches_to_tree(data):
    tree = BinaryTree(root_node_data=data[0])
    tree.cursor.add_left_child(data[1])
    tree.move_left()
    assert len(tree.cursor.data) == 200

def test_add_branch_to_tree(data):
    node = BinaryTree.Node(data=data[0], parent=None)
    assert node.is_leaf == True
    assert node.child_count == 0
    node.add_left_child(data[1])
    assert node.is_leaf == False
    assert node.child_count == 1
    node.add_right_child(data[0])
    assert node.is_leaf == False
    assert node.child_count == 2

@pytest.fixture
def small_tree():
    tree = BinaryTree(root_node_data="root")
    tree.cursor.add_left_child("hello")
    tree.cursor.add_right_child("hi")
    tree.move_left()
    tree.cursor.add_left_child("goodbye")
    tree.move_left()
    tree.cursor.add_right_child("hello again")
    tree.reset_cursor()
    return tree

def test_find_leaf_nodes(small_tree: BinaryTree):
    leaf_nodes = small_tree.get_leaf_nodes()
    leaf_data = [leaf.data for leaf in leaf_nodes]
    assert leaf_data == ["hello again", "hi"]
    
    
