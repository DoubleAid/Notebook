/*
 * @lc app=leetcode id=42 lang=cpp
 *
 * [42] Trapping Rain Water
 *
 * https://leetcode.com/problems/trapping-rain-water/description/
 *
 * algorithms
 * Hard (61.90%)
 * Likes:    31670
 * Dislikes: 505
 * Total Accepted:    2.1M
 * Total Submissions: 3.4M
 * Testcase Example:  '[0,1,0,2,1,0,1,3,2,1,2,1]'
 *
 * Given n non-negative integers representing an elevation map where the width
 * of each bar is 1, compute how much water it can trap after raining.
 * 
 * 
 * Example 1:
 * 
 * Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
 * Output: 6
 * Explanation: The above elevation map (black section) is represented by array
 * [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue
 * section) are being trapped.
 * 
 * 
 * Example 2:
 * 
 * Input: height = [4,2,0,3,2,5]
 * Output: 9
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == height.length
 * 1 <= n <= 2 * 10^4
 * 0 <= height[i] <= 10^5
 * 
 * 
 */
#include <algorithm>
#include <vector>

using namespace std;
// @lc code=start
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size()-1;
        int result = 0;
        while (left < right) {
            if (height[left] > height[right]) {
                int current_height = height[right];
                while (right > left && height[right] <= current_height) {
                    result += (current_height - height[right]);
                    right--;
                }
            } else {
                int current_height = height[left];
                while (left < right && height[left] <= current_height) {
                    result += (current_height - height[left]);
                    left++;
                }
            }
        }
        return result;
    }
};
// @lc code=end

// 322/322 cases passed (8 ms)
// Your runtime beats 72.65 % of cpp submissions
// Your memory usage beats 96.02 % of cpp submissions (22.2 MB)