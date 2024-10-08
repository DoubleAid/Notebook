/*
 * @lc app=leetcode id=44 lang=cpp
 *
 * [44] Wildcard Matching
 *
 * https://leetcode.com/problems/wildcard-matching/description/
 *
 * algorithms
 * Hard (27.89%)
 * Likes:    8149
 * Dislikes: 344
 * Total Accepted:    571.7K
 * Total Submissions: 2M
 * Testcase Example:  '"aa"\n"a"'
 *
 * Given an input string (s) and a pattern (p), implement wildcard pattern
 * matching with support for '?' and '*' where:
 * 
 * 
 * '?' Matches any single character.
 * '*' Matches any sequence of characters (including the empty sequence).
 * 
 * 
 * The matching should cover the entire input string (not partial).
 * 
 * 
 * Example 1:
 * 
 * Input: s = "aa", p = "a"
 * Output: false
 * Explanation: "a" does not match the entire string "aa".
 * 
 * 
 * Example 2:
 * 
 * Input: s = "aa", p = "*"
 * Output: true
 * Explanation: '*' matches any sequence.
 * 
 * 
 * Example 3:
 * 
 * Input: s = "cb", p = "?a"
 * Output: false
 * Explanation: '?' matches 'c', but the second letter is 'a', which does not
 * match 'b'.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s.length, p.length <= 2000
 * s contains only lowercase English letters.
 * p contains only lowercase English letters, '?' or '*'.
 * 
 * 
 */
#include <vector>

using namespace std;
// @lc code=start
class Solution {
public:
    bool isMatch(string s, string p) {
        vector<vector<bool>> dp(s.size()+1, vector<bool>(p.size()+1, false));
        dp[0][0] = true;
        int index = 0;
        while (index < p.size() && p[index++] == '*') dp[0][index] = true;
        for (int i = 1; i < s.size()+1; i++) {
            for (int j = 1; j < p.size()+1; j++) {
                if ((p[j-1] == '?' || p[j-1] == s[i-1]) && dp[i-1][j-1]) {
                    dp[i][j] = true;
                } else if (p[j-1] == '*' && (dp[i-1][j-1] || dp[i-1][j] || dp[i][j-1])) {
                    dp[i][j] = true;
                }
            }
        }
        return dp[s.size()][p.size()];
    }
};
// @lc code=end

// 1811/1811 cases passed (65 ms)
// Your runtime beats 41.67 % of cpp submissions
// Your memory usage beats 78.8 % of cpp submissions (15.4 MB)