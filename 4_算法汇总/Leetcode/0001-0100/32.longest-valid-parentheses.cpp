/*
 * @lc app=leetcode id=32 lang=cpp
 *
 * [32] Longest Valid Parentheses
 */

#include <algorithm>
#include <string>

using namespace std;

// @lc code=start
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxLength = 0;
        vector<int> count;
        count.push_back(-1);
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') {
                count.emplace_back(i);
            } else {
                count.pop_back();
                if (count.empty()) {
                    count.push_back(i);
                } else {
                    maxLength = max(maxLength, i - count.back());
                }
            }
        }
        return maxLength;
    }
};
// @lc code=end

// 231/231 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 38.43 % of cpp submissions (8.8 MB)