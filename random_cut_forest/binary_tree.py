from typing import List, Any, Type, Tuple

NodeType = Type["BinaryTree.Node"]

class BinaryTree:

    root_node: NodeType
    cursor: NodeType

    def __init__(self, root_node_data):
        self.root_node = BinaryTree.Node(data=root_node_data, parent=None)
        self.cursor = self.root_node

    def move_left(self):
        if self.cursor.left_child:
            self.cursor = self.cursor.left_child
        else:
            raise BinaryTree.Nodeless

    def move_right(self):
        if self.cursor.right_child:
            self.cursor = self.cursor.right_child
        else:
            raise BinaryTree.Nodeless

    def move_up(self):
        if self.cursor.parent:
            self.cursor = self.cursor.parent
        else:
            raise BinaryTree.Nodeless
        
    def reset_cursor(self):
        self.cursor = self.root_node
        
    def get_leaf_nodes(self, node: NodeType = None, so_far: List[NodeType] = []):

        if not node:
            node = self.root_node
        
        if node.is_leaf:
            return so_far + [node]
        else:
            if node.left_child:
                so_far = self.get_leaf_nodes(node.left_child, so_far)
            if node.right_child:
                so_far = self.get_leaf_nodes(node.right_child, so_far)

        return so_far


    class Node:

        data: Any
        parent: NodeType
        left_child: NodeType = None
        right_child: NodeType = None
        depth: int = 0

        def __init__(self, data: Any, parent: NodeType):
            self.data = data
            self.parent = parent
            
            if parent:
                self.depth = parent.depth + 1

        def add_left_child(self, data: Any):
            self.left_child = BinaryTree.Node(data=data, parent=self)
            
        def add_right_child(self, data: Any):
            self.right_child = BinaryTree.Node(data=data, parent=self)
        
        @property
        def child_count(self):
            return sum([self.left_child is not None, self.right_child is not None])
        
        @property
        def is_leaf(self):
            return self.child_count == 0
        
    class Nodeless(Exception):
        def __init__(self):
            super().__init__("Branch does not exist.")

