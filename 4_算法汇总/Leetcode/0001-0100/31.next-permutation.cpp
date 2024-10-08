/*
 * @lc app=leetcode id=31 lang=cpp
 *
 * [31] Next Permutation
 */

#include <algorithm>

using namespace std;

// @lc code=start
class Solution {
public:
    void nextPermutation(std::vector<int>& nums) {
        int n = nums.size();
        int i, j;
        
        // 1. 从后向前查找第一对连续的数字，使得 nums[i] < nums[i + 1]
        for (i = n - 2; i >= 0; --i) {
            if (nums[i] < nums[i + 1]) {
                break;
            }
        }

        if (i >= 0) { // 如果找到了这样的数字
            // 2. 从后向前查找第一个大于 nums[i] 的数字
            for (j = n - 1; j > i; --j) {
                if (nums[j] > nums[i]) {
                    break;
                }
            }
            // 3. 交换 nums[i] 和 nums[j]
            std::swap(nums[i], nums[j]);
        }

        // 4. 翻转从位置 i + 1 到末尾的部分
        std::reverse(nums.begin() + i + 1, nums.end());
    }
};
// @lc code=end

// 266/266 cases passed (8 ms)
// Your runtime beats 14.9 % of cpp submissions
// Your memory usage beats 54.11 % of cpp submissions (14.6 MB)

