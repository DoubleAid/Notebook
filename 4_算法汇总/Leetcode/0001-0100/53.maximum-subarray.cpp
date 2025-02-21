/*
 * @lc app=leetcode id=53 lang=cpp
 *
 * [53] Maximum Subarray
 *
 * https://leetcode.com/problems/maximum-subarray/description/
 *
 * algorithms
 * Medium (50.75%)
 * Likes:    33686
 * Dislikes: 1426
 * Total Accepted:    3.9M
 * Total Submissions: 7.7M
 * Testcase Example:  '[-2,1,-3,4,-1,2,1,-5,4]'
 *
 * Given an integer array nums, find the subarray with the largest sum, and
 * return its sum.
 * 
 * 
 * Example 1:
 * 
 * Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
 * Output: 6
 * Explanation: The subarray [4,-1,2,1] has the largest sum 6.
 * 
 * 
 * Example 2:
 * 
 * Input: nums = [1]
 * Output: 1
 * Explanation: The subarray [1] has the largest sum 1.
 * 
 * 
 * Example 3:
 * 
 * Input: nums = [5,4,-1,7,8]
 * Output: 23
 * Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 
 * 
 * 
 * Follow up: If you have figured out the O(n) solution, try coding another
 * solution using the divide and conquer approach, which is more subtle.
 * 
 */

#include <vector>

using namespace std;
// @lc code=start
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int>::iterator left = nums.begin(), right = nums.begin();
        int current_sum = *left;
        int max_sum = *left;
        while (right + 1 != nums.end()) {
            right++;
            if (current_sum <= 0) {
                left = right;
                current_sum = *left;
            } else {
                current_sum += *right;
            }
            if (current_sum > max_sum)
                max_sum = current_sum;
        }
        return max_sum;
    }
};
// @lc code=end

// 210/210 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 18.15 % of cpp submissions (71.8 MB)
