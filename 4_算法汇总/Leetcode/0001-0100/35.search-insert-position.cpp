/*
 * @lc app=leetcode id=35 lang=cpp
 *
 * [35] Search Insert Position
 */

// @lc code=start
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int start = 0, end = nums.size() - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        if (start == nums.size() - 1 && target > nums.back()) start++;
        return start;
    }
};
// @lc code=end

// 65/65 cases passed (3 ms)
// Your runtime beats 71.1 % of cpp submissions
// Your memory usage beats 7.55 % of cpp submissions (12.1 MB)