/*
 * @lc app=leetcode id=10 lang=cpp
 *
 * [10] Regular Expression Matching
 */
#include <cctype>
#include <string>
#include <vector>
using namespace std;

// @lc code=start
class Solution {
public:
    // 首先给一个递归的实现方法
    // bool isMatch(string s, string p) {
    //     if (p.empty() && s.empty()) return true;
    //     if (p.empty() && !s.empty()) return false;

    //     if (p.size() > 1 && p[1] == '*') {
    //         // 一个也不消
    //         if (isMatch(s, p.substr(2))) return true;
    //         if (!s.empty() && (p[0] == '.' || s[0] == p[0])) {
    //             // 消 n 个
    //             if (isMatch(s.substr(1), p)) return true;
    //         }
    //         return false;
    //     } else if (!s.empty() && (s[0] == p[0] || p[0] == '.')) {
    //         if (isMatch(s.substr(1), p.substr(1))) return true;
    //     }
    //     return false;
    // }
    // 可以解决，但超时了

    // 第二种 动态规划
    // 用 dp 二维数组表示 前面是否匹配
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
        dp[0][0] = true;

        for (int j = 2; j <= n; j++) {
            dp[0][j] = dp[0][j-2] && p[j-1] == '*';
        }

        for (int i = 1; i <= m; i ++) {
            for (int j = 1; j <= n; j++) {
                if (p[j-1] == '*') {
                    dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'));
                } else {
                    dp[i][j] = dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '.');
                }
            }
        }

        return dp[m][n];
    }
};
// @lc code=end

// 356/356 cases passed (3 ms)
// Your runtime beats 68.56 % of cpp submissions
// Your memory usage beats 52.91 % of cpp submissions (8.6 MB)