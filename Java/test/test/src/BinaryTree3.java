class Node {
    int item;
    Node left, right;

    public Node (int key){
        item = key;
        left = right = null;
    }
}

class BinaryTree3 {
    Node root;
    BinaryTree3() {
        root = null;
    }

    void Levelorder(){
        int h = height(root);
        int i;
        for (i = 0; i<h; i++){
            CurrentLevel(root, i);
        }
    }

    int height (Node root){
        if (root == null)
            return 0;
        else {
            int lheight = height(root.left);
            int rheight = height(root.right);

            if (lheight > rheight)
                return (lheight + 1);
            else
                return (rheight + 1);
        }
    }

    void CurrentLevel (Node root, int level){
        if (root == null)
            return;
        if (level == 1)
            System.out.print(root.item + " ");
        else if (level > 1){
            CurrentLevel(root.left, level -1);
            CurrentLevel(root.right, level-1);
        }
    }

    public static void main(String args[]){
        BinaryTree3 tree = new BinaryTree3();
        tree.root = new Node(1);
        tree.root.left = new Node(2);
        tree.root.right = new Node(3);
        tree.root.left.left = new Node(4);
        tree.root.left.right = new Node(5);
 
        System.out.println("Level order traversal of"
                           + " binary tree is ");
        tree.Levelorder();
    }
}
