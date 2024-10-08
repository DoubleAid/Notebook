/*
 * @lc app=leetcode id=26 lang=cpp
 *
 * [26] Remove Duplicates from Sorted Array
 */

// @lc code=start
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int duplicate = 0;
        while (duplicate+1 < nums.size() && nums[duplicate] != nums[duplicate+1]) duplicate++;
        int index = duplicate + 1;
        while (index < nums.size()) {
            if (nums[index] != nums[index-1]) {
                nums[duplicate+1] = nums[index];
                duplicate++;
            }
            index++;
        }
        return duplicate+1;
    }
};
// @lc code=end

// 362/362 cases passed (11 ms)
// Your runtime beats 35.38 % of cpp submissions
// Your memory usage beats 91.41 % of cpp submissions (21 MB)