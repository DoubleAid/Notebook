/*
 * @lc app=leetcode id=47 lang=cpp
 *
 * [47] Permutations II
 *
 * https://leetcode.com/problems/permutations-ii/description/
 *
 * algorithms
 * Medium (59.15%)
 * Likes:    8447
 * Dislikes: 142
 * Total Accepted:    929.4K
 * Total Submissions: 1.6M
 * Testcase Example:  '[1,1,2]'
 *
 * Given a collection of numbers, nums, that might contain duplicates, return
 * all possible unique permutations in any order.
 * 
 * 
 * Example 1:
 * 
 * Input: nums = [1,1,2]
 * Output:
 * [[1,1,2],
 * ⁠[1,2,1],
 * ⁠[2,1,1]]
 * 
 * 
 * Example 2:
 * 
 * Input: nums = [1,2,3]
 * Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 8
 * -10 <= nums[i] <= 10
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    void backtrack(const std::vector<int>& nums, std::vector<bool>& used, std::vector<int>& path, std::vector<std::vector<int>>& result) {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        
        for (int i = 0; i < nums.size(); ++i) {
            // Skip used elements and duplicates (only skip duplicates if the previous one is not used)
            if (used[i] || (i > 0 && nums[i] == nums[i-1] && !used[i-1])) {
                continue;
            }
            
            // Include this element in the permutation
            used[i] = true;
            path.push_back(nums[i]);
            
            // Continue to permute with the current element included
            backtrack(nums, used, path, result);
            
            // Backtrack, remove the element from the current permutation
            path.pop_back();
            used[i] = false;
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        std::vector<std::vector<int>> result;
        std::vector<bool> used(nums.size(), false);
        std::vector<int> path;
        std::sort(nums.begin(), nums.end());
        backtrack(nums, used, path, result);
        return result;
    }
};
// @lc code=end

// 33/33 cases passed (3 ms)
// Your runtime beats 89.06 % of cpp submissions
// Your memory usage beats 84.55 % of cpp submissions (10.7 MB)