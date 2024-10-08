/*
 * @lc app=leetcode id=16 lang=cpp
 *
 * [16] 3Sum Closest
 */

// @lc code=start
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int res = nums[0] + nums[1] + nums[2];
        int min_abs = abs(res - target);
        for (int i=0; i < nums.size(); i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                int tmp_abs = abs(sum - target);
                if (tmp_abs < min_abs) {
                    res = sum;
                    min_abs = tmp_abs;
                }
                if (sum < target) {
                    left++;
                } else if (sum > target) {
                    right--;
                } else {
                    return target;
                }
            }
        }
        return res;
    }
};
// @lc code=end

// 102/102 cases passed (13 ms)
// Your runtime beats 81.85 % of cpp submissions
// Your memory usage beats 22.93 % of cpp submissions (12.9 MB)