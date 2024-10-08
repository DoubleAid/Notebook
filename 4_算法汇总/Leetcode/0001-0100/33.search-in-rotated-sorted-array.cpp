/*
 * @lc app=leetcode id=33 lang=cpp
 *
 * [33] Search in Rotated Sorted Array
 */

// @lc code=start
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int start = 0, end = nums.size() - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] >= nums[start]) {
                if (target < nums[mid] && target >= nums[start]) {
                    end = mid;
                } else {
                    start = mid + 1;
                }
            } else {
                if (target >= nums[start] || target < nums[mid]) {
                    end = mid;
                } else {
                    start = mid + 1;
                }
            }
        }
        if (nums[start] == target) return start;
        return -1;
    }
};
// @lc code=end

// 195/195 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 35.2 % of cpp submissions (13.5 MB)