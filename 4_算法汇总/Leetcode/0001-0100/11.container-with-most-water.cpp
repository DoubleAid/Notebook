/*
 * @lc app=leetcode id=11 lang=cpp
 *
 * [11] Container With Most Water
 */
#include <vector>
#include <algorithm>
using namespace std;

// @lc code=start
class Solution {
public:
    int maxArea(vector<int>& height) {
        int max_area = 0;
        int left = 0;
        int right = height.size() - 1;

        while (left < right) {
            int h = min(height[left], height[right]);
            max_area = max(max_area, h * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return max_area;
    }
};
// @lc code=end

