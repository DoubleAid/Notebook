/*
 * @lc app=leetcode id=91 lang=cpp
 *
 * [91] Decode Ways
 */

#include <algorithm>
#include <string>
#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    int numDecodings(string s) {
        vector<int> dp(s.size()+1, 0);
        dp[0] = 1;
        for (int i = 1; i <= s.size(); i++) {
            if (dp[i-1] != 0 && s[i-1] != '0') {
                dp[i] += dp[i-1];
            }
            if (i-2 >= 0 && dp[i-2] != 0 && s[i-2] != '0' && stoi(s.substr(i-2, 2)) <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp.back();
    }


};
// @lc code=end

// 269/269 cases passed (4 ms)
// Your runtime beats 31.34 % of cpp submissions
// Your memory usage beats 43.67 % of cpp submissions (8.4 MB)