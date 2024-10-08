/*
 * @lc app=leetcode id=41 lang=cpp
 *
 * [41] First Missing Positive
 *
 * https://leetcode.com/problems/first-missing-positive/description/
 *
 * algorithms
 * Hard (39.45%)
 * Likes:    16662
 * Dislikes: 1838
 * Total Accepted:    1.2M
 * Total Submissions: 3.1M
 * Testcase Example:  '[1,2,0]'
 *
 * Given an unsorted integer array nums. Return the smallest positive integer
 * that is not present in nums.
 * 
 * You must implement an algorithm that runs in O(n) time and uses O(1)
 * auxiliary space.
 * 
 * 
 * Example 1:
 * 
 * Input: nums = [1,2,0]
 * Output: 3
 * Explanation: The numbers in the range [1,2] are all in the array.
 * 
 * 
 * Example 2:
 * 
 * Input: nums = [3,4,-1,1]
 * Output: 2
 * Explanation: 1 is in the array but 2 is missing.
 * 
 * 
 * Example 3:
 * 
 * Input: nums = [7,8,9,11,12]
 * Output: 1
 * Explanation: The smallest positive integer 1 is missing.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -2^31 <= nums[i] <= 2^31 - 1
 * 
 * 
 */
#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        bool is_found = false;
        for (int num : nums) {
            if (num == 1) {
                is_found = true;
            }
        }
        if (!is_found) return 1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] <= 0 || nums[i] > nums.size()) nums[i] = 1;
        }
        for (int i = 0; i < nums.size(); i++) {
            int index = abs(nums[i]) - 1;
            if (nums[index] > 0) nums[index] = -nums[index];
        }
        int result = 1;
        for (int num: nums) {
            if (num < 0) {
                result++;
            } else {
                break;
            }
        }
        return result;
    }
};
// @lc code=end

// 177/177 cases passed (55 ms)
// Your runtime beats 29.6 % of cpp submissions
// Your memory usage beats 22.05 % of cpp submissions (53.5 MB)