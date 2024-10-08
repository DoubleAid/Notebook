/*
 * @lc app=leetcode id=22 lang=cpp
 *
 * [22] Generate Parentheses
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// @lc code=start
class Solution {
public:
vector<string> generateParenthesis(int n) {
    vector<vector<string>> dp(n + 1);
    dp[0] = {""};  // Base case: an empty string

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < i; j++) {
            for (const string& left : dp[j]) {          // All combinations from dp[j]
                for (const string& right : dp[i - j - 1]) {  // All combinations from dp[i-j-1]
                    dp[i].push_back("(" + left + ")" + right);
                }
            }
        }
    }

    return dp[n];
}
};
// @lc code=end

// 8/8 cases passed (5 ms)
// Your runtime beats 30.47 % of cpp submissions
// Your memory usage beats 98.7 % of cpp submissions (9.1 MB)