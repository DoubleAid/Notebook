/*
 * @lc app=leetcode id=15 lang=cpp
 *
 * [15] 3Sum
 */
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size()-2; i++) {
            if (i>0&&nums[i] == nums[i-1]) continue;
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0) {
                    left++;
                } else if (sum > 0) {
                    right--;
                } else {
                    res.emplace_back(vector<int> {nums[i], nums[left], nums[right]});
                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
                    left++;
                    right--;
                }
            }
        }
        return res;
    }
};
// @lc code=end

// 313/313 cases passed (69 ms)
// Your runtime beats 55.26 % of cpp submissions
// Your memory usage beats 94.28 % of cpp submissions (27.1 MB)

// 有必要可以先排序