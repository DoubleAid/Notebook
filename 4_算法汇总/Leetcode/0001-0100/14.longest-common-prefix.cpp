/*
 * @lc app=leetcode id=14 lang=cpp
 *
 * [14] Longest Common Prefix
 */
#include <string>
#include <algorithm>
#include <vector>
using namespace std;
// @lc code=start
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int p = 0;
        while (true) {
            if (p >= strs[0].size()) return strs[0].substr(0, p);
            char m = strs[0][p];
            for (int i = 1; i < strs.size(); i++) {
                if (p >= strs[i].size() || strs[i][p] != m) return strs[0].substr(0, p);
            }
            p++;
        } 
    }
};
// @lc code=end

// 125/125 cases passed (7 ms)
// Your runtime beats 28.4 % of cpp submissions
// Your memory usage beats 88.33 % of cpp submissions (10.8 MB)