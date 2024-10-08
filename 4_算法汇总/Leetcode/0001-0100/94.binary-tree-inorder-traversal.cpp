/*
 * @lc app=leetcode id=94 lang=cpp
 *
 * [94] Binary Tree Inorder Traversal
 */

#include <vector>

using namespace std;

class TreeNode;
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
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        TreeNode *current = root, *prev = nullptr;
        while (current != nullptr) {
            if (current->left != nullptr) {
                prev = current->left;
                while (prev->right != nullptr && prev->right != current) {
                    prev = prev->right;
                }
                if (prev->right == nullptr) {
                    prev->right = current;
                    current = current->left;
                } else {
                    prev->right = nullptr;
                    res.emplace_back(current->val);
                    current = current->right;
                }
            } else {
                res.emplace_back(current->val);
                current = current->right;
            }
        }
        return res;
    }
};
// @lc code=end

// 70/70 cases passed (4 ms)
// Your runtime beats 31.87 % of cpp submissions
// Your memory usage beats 72.59 % of cpp submissions (9.8 MB)