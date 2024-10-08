/*
 * @lc app=leetcode id=100 lang=cpp
 *
 * [100] Same Tree
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
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) return true;
        if (p == nullptr || q == nullptr) return false;
        if (p->val == q->val) {
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }
        return false;
    }
};
// @lc code=end

// 66/66 cases passed (3 ms)
// Your runtime beats 32.07 % of cpp submissions
// Your memory usage beats 79.86 % of cpp submissions (11.8 MB)