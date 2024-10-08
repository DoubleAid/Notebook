/*
 * @lc app=leetcode id=40 lang=cpp
 *
 * [40] Combination Sum II
 *
 * https://leetcode.com/problems/combination-sum-ii/description/
 *
 * algorithms
 * Medium (54.47%)
 * Likes:    10317
 * Dislikes: 290
 * Total Accepted:    974.3K
 * Total Submissions: 1.8M
 * Testcase Example:  '[10,1,2,7,6,1,5]\n8'
 *
 * Given a collection of candidate numbers (candidates) and a target number
 * (target), find all unique combinations in candidates where the candidate
 * numbers sum to target.
 * 
 * Each number in candidates may only be used once in the combination.
 * 
 * Note: The solution set must not contain duplicate combinations.
 * 
 * 
 * Example 1:
 * 
 * Input: candidates = [10,1,2,7,6,1,5], target = 8
 * Output: 
 * [
 * [1,1,6],
 * [1,2,5],
 * [1,7],
 * [2,6]
 * ]
 * 
 * 
 * Example 2:
 * 
 * Input: candidates = [2,5,2,1,2], target = 5
 * Output: 
 * [
 * [1,2,2],
 * [5]
 * ]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= candidates.length <= 100
 * 1 <= candidates[i] <= 50
 * 1 <= target <= 30
 * 
 * 
 */

#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        sort(candidates.begin(), candidates.end());
        if (candidates.size() == 0) return result;
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates[i] > target) break;
            if (i > 0 && candidates[i] == candidates[i-1]) continue;
            if (candidates[i] == target) {
                result.emplace_back(vector<int>{target});
                return result;
            }
            vector<int> next_candidates(candidates.begin()+i+1, candidates.end());
            vector<vector<int>> res = combinationSum2(next_candidates, target - candidates[i]);
            for (vector<int> k : res) {
                k.insert(k.begin(), candidates[i]);
                result.emplace_back(k);
            }
        }
        return result;
    }
};
// @lc code=end

// 176/176 cases passed (3 ms)
// Your runtime beats 79.22 % of cpp submissions
// Your memory usage beats 10.83 % of cpp submissions (18.1 MB)