/*
 * @lc app=leetcode id=95 lang=cpp
 *
 * [95] Unique Binary Search Trees II
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
    vector<TreeNode*> generateTrees(int n) {
        return generater(1, n);
    }

    vector<TreeNode*>generater(int start, int end) {
        vector<TreeNode*> res;
        if (start > end) {
            res.push_back(nullptr);
            return res;
        }
        for (int i = start; i <= end; i++) {
            vector<TreeNode*> leftTrees = generater(start, i-1);
            vector<TreeNode*> rightTrees = generater(i+1, end);
            for (TreeNode* left: leftTrees) {
                for (TreeNode* right: rightTrees) {
                    TreeNode* node = new TreeNode(i, left, right);
                    res.push_back(node);
                }
            }
        }
        return res;
    }
};
// @lc code=end

// 8/8 cases passed (10 ms)
// Your runtime beats 67.8 % of cpp submissions
// Your memory usage beats 39.25 % of cpp submissions (19.1 MB)