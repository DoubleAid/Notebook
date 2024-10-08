/*
 * @lc app=leetcode id=98 lang=cpp
 *
 * [98] Validate Binary Search Tree
 */

// @lc code=start
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */

// Morris遍历的步骤如下：

// 初始化：从根节点开始。
// 寻找前驱：
// 对于每个节点，如果它的左子节点不为空，则寻找这个左子节点的最右侧节点，即这个节点的前驱。
// 如果前驱节点的右指针为空，将它的右指针设置为当前节点，然后当前节点移动到它的左子节点。
// 如果前驱节点的右指针已经指向当前节点（表示线索已经创建），则断开这个线索（将前驱节点的右指针恢复为NULL），表示左子树已经遍历完成。
// 遍历右子树：
// 当前节点向右移动，继续进行中序遍历。
// 重复以上步骤，直到所有节点都被访问。

class Solution {
public:
    // 方法一
    bool isValidBST(TreeNode* root) {
        TreeNode *prev = nullptr, *temp = nullptr;
        while (root != nullptr) {
            if (root->left != nullptr) {
                // 如果它的左子节点不为空，则寻找这个左子节点的最右侧节点，即这个节点的前驱。
                temp = root->left;
                while (temp->right != nullptr && temp->right != root) {
                    temp = temp->right;
                }
                // 第一次
                if (temp->right == nullptr) {
                    temp->right = root;
                    root = root->left;
                } else {
                // 第二次
                    if (prev != nullptr && prev->val > root->val) return false;
                    temp->right = nullptr;
                    prev = root;
                    root = root->right;   
                }
            } else {
                if (prev != nullptr && prev->val > root->val) return false;
                prev = root;
                root = root->right;
            }
        }
        return true;
    }

    // 方法二
    bool isValidBSTHelper(TreeNode* node, long long minVal, long long maxVal) {
        if (!node) return true;  // 空树是BST
        if (node->val <= minVal || node->val >= maxVal) return false;  // 超出范围，不是BST
        return isValidBSTHelper(node->left, minVal, node->val) && isValidBSTHelper(node->right, node->val, maxVal);
    }

    bool isValidBST(TreeNode* root) {
        return isValidBSTHelper(root, LLONG_MIN, LLONG_MAX);
    }
};
// @lc code=end

