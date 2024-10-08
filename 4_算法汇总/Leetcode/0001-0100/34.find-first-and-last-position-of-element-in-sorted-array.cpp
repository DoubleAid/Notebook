/*
 * @lc app=leetcode id=34 lang=cpp
 *
 * [34] Find First and Last Position of Element in Sorted Array
 */

// @lc code=start
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> result = {-1, -1};
        if (nums.empty()) return result;
        int start = 0, end = nums.size() - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                int left = mid, right = mid;
                while (left > 0 && nums[left-1] == target) left--;
                while (right < nums.size()-1 && nums[right+1] == target) right++;
                return vector<int>{left, right};
            } else if (nums[mid] > target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        if (nums[start] == target) {
            int left = start, right = start;
            while (left > 0 && nums[left-1] == target) left--;
            while (right < nums.size()-1 && nums[right+1] == target) right++;
            return vector<int>{left, right};
        }
        return result;
    }
};
// @lc code=end

// 88/88 cases passed (3 ms)
// Your runtime beats 89.23 % of cpp submissions
// Your memory usage beats 86.1 % of cpp submissions (16 MB)