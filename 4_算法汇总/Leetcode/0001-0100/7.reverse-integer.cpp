/*
 * @lc app=leetcode id=7 lang=cpp
 *
 * [7] Reverse Integer
 */

// @lc code=start
class Solution {
public:
    int reverse(int x) {
        long res = 0;
        while (x != 0) {
            res = res * 10 + x % 10;
            x = x / 10;
            if (res > INT_MAX || res < INT_MIN) return 0;
        }
        return int(res);
    }
};
// @lc code=end

// 1045/1045 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 16.04 % of cpp submissions (7.5 MB)