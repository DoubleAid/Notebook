/*
 * @lc app=leetcode id=4 lang=cpp
 *
 * [4] Median of Two Sorted Arrays
 */

// @lc code=start
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {


        // 解法和上学时的消元类似
        // 首先判定奇偶性，
        int len1 = nums1.size(), len2 = nums2.size();
        if (len1 > len2) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int total_left = (len1 + len2 + 1) / 2;
        bool is_odd = (len1 + len2) % 2 == 1;
        // i 和 j 的和应该是长度的一半 0 1 2 3 4 5 6 7 8 9 10 11
        // m = 5, n = 6 即一个奇数，一个偶数 共11个，i + j 应该是 6
        //      假设 i = 3， j = 2， 那么中间数就是 min(nums1[i], nums2[j])
        // m = 5, n = 7 即两个奇数， 共12个，i + j 应该是 应该是 6
        //      假设 i = 3， j = 3，那么中间数就是 max(nums1[i-1]+nums2[j-1]) + min(nums1[i], nums2[j]) / 2
        // 两个偶数情况和上面相同
        // 考虑几种特殊情况
        int left = 0, right = len1;
        while (left < right) {
            int i = left + (right - left) / 2;
            int j = total_left - i;
            if (nums1[i] < nums2[j-1]) {
                // i 需要向右移动
                left = i + 1;
            } else {
                right = i;
            }
        }
        int i = left;
        int j = total_left - i;
        int nums1LeftMax = i == 0? INT_MIN : nums1[i-1];
        int nums2LeftMax = j == 0? INT_MIN : nums2[j-1];
        int left_max = max(nums1LeftMax, nums2LeftMax);
        if (is_odd) {
            return left_max;
        } else {
            int nums1RightMin = i == len1? INT_MAX: nums1[i];
            int nums2RightMin = j == len2? INT_MAX: nums2[j];
            int right_min = min(nums1RightMin, nums2RightMin);
            return (left_max + right_min) / 2.0;
        }
    }
};
// @lc code=end

// 2094/2094 cases passed (11 ms)
// Your runtime beats 98.86 % of cpp submissions
// Your memory usage beats 61.73 % of cpp submissions (94.6 MB)