/*
 * @lc app=leetcode id=29 lang=cpp
 *
 * [29] Divide Two Integers
 */

#include <iostream>

using namespace std;

// @lc code=start
class Solution {
public:
    int divide(int dividend, int divisor) {
        if (dividend == INT_MIN && divisor == -1) return INT_MAX;
        int flag = (dividend > 0)^(divisor > 0) ? -1 : 1;
        long long n = abs((long long) dividend);
        long long d = abs((long long) divisor);
        long long result = 0;

        while (n >= d) {
            long long temp = d, multiple = 1;
            while (n >= (temp << 1)) {
                temp <<= 1;
                multiple <<= 1;
            }
            n -= temp;
            result += multiple;
        }
        return flag > 0? result: -result;
    }
};
// @lc code=end

// 994/994 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 55.48 % of cpp submissions (7.5 MB)