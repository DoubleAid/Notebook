/*
 * @lc app=leetcode id=39 lang=cpp
 *
 * [39] Combination Sum
 */

#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        if (target == 1 || candidates.size() <= 0) return result;
        int index = 0;
        while (index < candidates.size() && candidates[index] > target) index++;
        if (index >= candidates.size()) return result;
        for (int i = index; i < candidates.size(); i++) {
            if (candidates[i] == target) {
                result.push_back({candidates[i]}); // 直接添加满足条件的单个数字
            } else {
                vector<int> next_candidates(candidates.begin() + i, candidates.end());
                vector<vector<int>> tmp = combinationSum(next_candidates, target - candidates[i]);
                for (vector<int>& vec : tmp) {
                    vec.insert(vec.begin(), candidates[i]);
                    result.push_back(vec);
                }
            }
        }
        return result;
    }
};
// @lc code=end

// 160/160 cases passed (17 ms)
// Your runtime beats 28.47 % of cpp submissions
// Your memory usage beats 29.4 % of cpp submissions (17.9 MB)