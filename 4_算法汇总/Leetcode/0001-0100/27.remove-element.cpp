/*
 * @lc app=leetcode id=27 lang=cpp
 *
 * [27] Remove Element
 */

// @lc code=start
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int k = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != val) {
                nums[k++] = nums[i];
            }
        }
        return k;
    }
};
// @lc code=end

// 114/114 cases passed (5 ms)
// Your runtime beats 27.68 % of cpp submissions
// Your memory usage beats 79.52 % of cpp submissions (10.4 MB)