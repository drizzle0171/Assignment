class Node {
    int item;
    Node left, right;

    public Node (int key){
        item = key;
        left = right = null;
    }
}

class BinaryTree2{
    Node root;
    BinaryTree2() {
        root = null;
    }

    void Postorder(Node node){
        if (node == null){
            return ;
        }
        Postorder(node.left);
        Postorder(node.right);
        System.out.print(node.item + " ");
    }

    void Postorder() {
        Postorder(root);
    }

    public static void main(String[] args){
        BinaryTree2 tree = new BinaryTree2();
        tree.root = new Node(1);
        tree.root.left = new Node(2);
        tree.root.right = new Node(3);
        tree.root.left.left = new Node(4);
        tree.root.left.right = new Node(5);
        
        System.out.println("\nPostorder traversal of binary tree is ");
        tree.Postorder();
    }
}