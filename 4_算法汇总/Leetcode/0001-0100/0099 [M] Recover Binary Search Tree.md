# 0099 [M] Recover Binary Search Tree
You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.

#### Example 1:
```
Input: root = [1,3,null,null,2]
Output: [3,1,null,null,2]
Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.
```
#### Example 2:
```
Input: root = [3,1,4,null,null,2]
Output: [2,1,4,null,null,3]
Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.
```

#### Constraints:
+ The number of nodes in the tree is in the range [2, 1000].
+ -231 <= Node.val <= 231 - 1

**Follow up:** A solution using O(n) space is pretty straight-forward. Could you devise a constant O(1) space solution?

#### 思路
第一种想法是先遍历一遍树上面的各个点， 记录下各个点的值， 对点进行排序，然后再放进树内即可

第二种想法是树内的大部分都是有序的， 用一个有序数列为例： `1 2 3 4 5 6 7 8`, 挑选其中的两个位置交换一下: `1 2 7 4 5 6 3 8`, 那么我们再从遍历一遍， 第一个矛盾点 是 `7 4`, 最后一个 `6 3`, 那么我们只要找到这两个矛盾点， 然后取第一个点的 第一个位置， 最后一个位置的后一个位置，交换一下就可以了
```C++
class Solution {
public:class Solution {
public:
    TreeNode *pre = NULL, *first = NULL, *last = NULL;
    void recoverTree(TreeNode* root) {
        inorder(root);
        swap(first->val, last->val);
    }

    void inorder(TreeNode* root) {
        if (!root) return;
        inorder(root->left);
        if (pre) {
            if (pre->val > root->val) {
                if (!first) first = pre;
                last = root;
            }
        }
        pre = root;
        inorder(root->right);
    }
};
        swap(first->val, last->val);
    }

    void inorder(TreeNode* root) {
        if (!root) return;
        inorder(root->left);
        if (pre) {
            if (pre->val > root->val) {
                if (!first) first = pre;
                last = root;
            }
        }
        pre = root;
        inorder(root->right);
    }
};
```