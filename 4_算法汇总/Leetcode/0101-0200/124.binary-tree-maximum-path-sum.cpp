/*
 * @lc app=leetcode id=124 lang=cpp
 *
 * [124] Binary Tree Maximum Path Sum
 *
 * https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
 *
 * algorithms
 * Hard (40.01%)
 * Likes:    16968
 * Dislikes: 749
 * Total Accepted:    1.4M
 * Total Submissions: 3.4M
 * Testcase Example:  '[1,2,3]'
 *
 * A path in a binary tree is a sequence of nodes where each pair of adjacent
 * nodes in the sequence has an edge connecting them. A node can only appear in
 * the sequence at most once. Note that the path does not need to pass through
 * the root.
 * 
 * The path sum of a path is the sum of the node's values in the path.
 * 
 * Given the root of a binary tree, return the maximum path sum of any
 * non-empty path.
 * 
 * 
 * Example 1:
 * 
 * Input: root = [1,2,3]
 * Output: 6
 * Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 =
 * 6.
 * 
 * 
 * Example 2:
 * 
 * Input: root = [-10,9,20,null,null,15,7]
 * Output: 42
 * Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 +
 * 7 = 42.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in the tree is in the range [1, 3 * 10^4].
 * -1000 <= Node.val <= 1000
 * 
 * 
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
#include <vector>
#include <utility>

using namespace std;

class Solution {
public:
    int maxPathSum(TreeNode* root) {
        return getMaxPathSum(root).first;
    }

    // 返回值是一个pair，第一个值是全路径的最大值，第二个值是以根节点为一端的最大路径值
    pair<int, int> getMaxPathSum(TreeNode* root) {
        if (root == nullptr) {
            return {INT_MIN, 0};
        }

        auto left = getMaxPathSum(root->left);
        auto right = getMaxPathSum(root->right);

        int singlePathMax = max(root->val, max(left.second, right.second) + root->val);
        int fullPathMax = max(left.first, right.first);
        fullPathMax = max(fullPathMax, max(left.second + root->val + right.second, singlePathMax));
        return {fullPathMax, singlePathMax};
    }
};
// @lc code=end