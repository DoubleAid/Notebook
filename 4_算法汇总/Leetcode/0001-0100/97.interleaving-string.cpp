/*
 * @lc app=leetcode id=97 lang=cpp
 *
 * [97] Interleaving String
 */
#include <vector>
#include <string>

using namespace std;

// 动态规划是解决这类字符串交错问题的一种非常有效的方法。你可以使用一个二维的动态规划数组 dp[i][j]，其中 dp[i][j] 表示 s1 的前 i 个字符和 s2 的前 j 个字符是否能交错组成 s3 的前 i+j 个字符。

// 定义动态规划数组如下：

// dp[0][0] = true，因为两个空字符串可以组成一个空字符串。
// dp[i][0] 表示只用 s1 的前 i 个字符是否能形成 s3 的前 i 个字符，如果 s1[0..i-1] 等于 s3[0..i-1]，则为 true。
// dp[0][j] 表示只用 s2 的前 j 个字符是否能形成 s3 的前 j 个字符，如果 s2[0..j-1] 等于 s3[0..j-1]，则为 true。
// 对于其他情况，动态规划转移方程为：

// dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
// 这个方程的意思是：

// 如果 s1 的第 i 个字符和 s3 的第 i+j 个字符相同，并且去掉这个字符的子问题 dp[i-1][j] 是 true 的话，那么 dp[i][j] 也是 true。
// 同理，如果 s2 的第 j 个字符和 s3 的第 i+j 个字符相同，并且去掉这个字符的子问题 dp[i][j-1] 是 true 的话，那么 dp[i][j] 也是 true。
// 最后，dp[len(s1)][len(s2)] 的值将告诉我们是否可以用 s1 和 s2 交错形成 s3。

// @lc code=start
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int s1_len = s1.size(), s2_len = s2.size(), s3_len = s3.size();
        if (s1_len + s2_len != s3_len) return false;
        vector<vector<bool>> dp(s2_len+1, vector<bool>(s1_len+1, false));
        dp[0][0] = true;
        
        int i = 1;
        while (i <= s1_len && dp[0][i-1] && s1[i-1] == s3[i-1]) dp[0][i++] = true;

        int j = 1;
        while (j <= s2_len && dp[j-1][0] && s2[j-1] == s3[j-1]) dp[j++][0] = true;

        for (int j = 1; j < s2_len+1; j++) {
            for (int i = 1; i < s1_len+1; i++) {
                dp[j][i] = (dp[j][i-1] && s1[i-1] == s3[i+j-1]) || (dp[j-1][i] && s2[j-1] == s3[i+j-1]);
            }
        }

        return dp[s2_len][s1_len];
    }
};
// @lc code=end

// 106/106 cases passed (6 ms)
// Your runtime beats 45.67 % of cpp submissions
// Your memory usage beats 79.25 % of cpp submissions (8.1 MB)