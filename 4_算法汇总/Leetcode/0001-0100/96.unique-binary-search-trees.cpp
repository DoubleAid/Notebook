/*
 * @lc app=leetcode id=96 lang=cpp
 *
 * [96] Unique Binary Search Trees
 */

#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(max(n+1, 3), 0);
        dp[0] = 1; dp[1] = 1; dp[2] = 2;
        if (n <= 2) return dp[n];
        for (int i = 3; i <= n; i++) {
            int count = 0;
            for(int j = 0; j < i; j++) {
                count += dp[j] * dp[i-j-1];
            }
            dp[i] = count;
        }
        return dp[n];
    }
};
// @lc code=end

// 典型的动态规划问题

// 19/19 cases passed (2 ms)
// Your runtime beats 59.42 % of cpp submissions
// Your memory usage beats 82.83 % of cpp submissions (7.1 MB)