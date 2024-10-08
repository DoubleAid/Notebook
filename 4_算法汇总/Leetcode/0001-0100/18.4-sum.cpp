/*
 * @lc app=leetcode id=18 lang=cpp
 *
 * [18] 4Sum
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res = {};
        if(nums.size() < 4) return res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size()-2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            for (int j = i + 1; j < nums.size()-1; j++) {
                if (j > i+1 && nums[j] == nums[j-1]) continue;
                int left = j+1, right = nums.size()-1;
                while (left < right) {
                    // 这一行要对第一个数字强转
                    long long sum = static_cast<long long>(nums[i]) + nums[j] + nums[left] + nums[right];
                    if (sum < target) {
                        left++;
                    } else if (sum > target) {
                        right--;
                    } else {
                        res.emplace_back(vector<int>{nums[i], nums[j], nums[left], nums[right]});
                        while (left < right && nums[left] == nums[left+1]) left++;
                        while (left < right && nums[right] == nums[right-1]) right--;
                        left++;
                        right--;
                    }
                }
            }
        }
        return res;
    }
};
// @lc code=end

// 294/294 cases passed (22 ms)
// Your runtime beats 81.03 % of cpp submissions
// Your memory usage beats 96.47 % of cpp submissions (15.7 MB)