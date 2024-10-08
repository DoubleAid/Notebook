/*
 * @lc app=leetcode id=8 lang=cpp
 *
 * [8] String to Integer (atoi)
 */

// @lc code=start
class Solution {
public:
    int myAtoi(string s) {
        int index = 0;
        int flag = 1;
        long res = 0;
        while (index < s.size() && s[index] == ' ') index++;
        if (index < s.size() && (s[index] == '-' || s[index] == '+')) {
            if (s[index] == '-') flag = -1; 
            index++;
        }
        while (index < s.size() && s[index] >= '0' && s[index] <= '9') {
            res = res * 10 + s[index] - '0';
            if (flag > 0 && res > INT_MAX) return INT_MAX;
            if (flag < 0 && -res < INT_MIN) return INT_MIN;
            index++;
        }
        return flag * res;
    }
};
// @lc code=end

// 1092/1092 cases passed (7 ms)
// Your runtime beats 12.18 % of cpp submissions
// Your memory usage beats 66.92 % of cpp submissions (8.2 MB)