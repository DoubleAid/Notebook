/*
 * @lc app=leetcode id=46 lang=cpp
 *
 * [46] Permutations
 *
 * https://leetcode.com/problems/permutations/description/
 *
 * algorithms
 * Medium (78.27%)
 * Likes:    18833
 * Dislikes: 322
 * Total Accepted:    2.1M
 * Total Submissions: 2.6M
 * Testcase Example:  '[1,2,3]'
 *
 * Given an array nums of distinct integers, return all the possible
 * permutations. You can return the answer in any order.
 * 
 * 
 * Example 1:
 * Input: nums = [1,2,3]
 * Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * Example 2:
 * Input: nums = [0,1]
 * Output: [[0,1],[1,0]]
 * Example 3:
 * Input: nums = [1]
 * Output: [[1]]
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 6
 * -10 <= nums[i] <= 10
 * All the integers of nums are unique.
 * 
 * 
 */
#include <vector>
#include <algorithm>

using namespace std;


// @lc code=start
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        if (nums.size() == 0) return vector<vector<int>>{{}};
        for (int i = 0; i < nums.size(); i++) {
            vector<int> candidates = nums;
            candidates.erase(candidates.begin()+i);
            vector<vector<int>> ans = permute(candidates);
            for (vector<int> a : ans) {
                a.insert(a.begin(), nums[i]);
                result.emplace_back(a);
            }
        }
        return result;
    }
};
// @lc code=end

// 26/26 cases passed (7 ms)
// Your runtime beats 12.19 % of cpp submissions
// Your memory usage beats 5.49 % of cpp submissions (13.7 MB)