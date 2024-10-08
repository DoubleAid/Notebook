/*
 * @lc app=leetcode id=50 lang=cpp
 *
 * [50] Pow(x, n)
 *
 * https://leetcode.com/problems/powx-n/description/
 *
 * algorithms
 * Medium (34.75%)
 * Likes:    9549
 * Dislikes: 9381
 * Total Accepted:    1.7M
 * Total Submissions: 4.8M
 * Testcase Example:  '2.00000\n10'
 *
 * Implement pow(x, n), which calculates x raised to the power n (i.e.,
 * x^n).
 * 
 * 
 * Example 1:
 * 
 * Input: x = 2.00000, n = 10
 * Output: 1024.00000
 * 
 * 
 * Example 2:
 * 
 * Input: x = 2.10000, n = 3
 * Output: 9.26100
 * 
 * 
 * Example 3:
 * 
 * Input: x = 2.00000, n = -2
 * Output: 0.25000
 * Explanation: 2^-2 = 1/2^2 = 1/4 = 0.25
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * -100.0 < x < 100.0
 * -2^31 <= n <= 2^31-1
 * n is an integer.
 * Either x is not zero or n > 0.
 * -10^4 <= x^n <= 10^4
 * 
 * 
 */
#include <vector>

using namespace std;
// @lc code=start
class Solution {
public:
    double myPow(double x, int n) {
        if (n == 0 || x == 1) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;
        int tmp = n / 2;
        if (n < 0) {
            x = 1/x;
            tmp = -(n / 2);
        }
        double val = myPow(x, tmp);
        val = val * val;
        if (n % 2 != 0) {
            val *= x;
        }
        return val;
    }
};
// @lc code=end

// 307/307 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 21.28 % of cpp submissions (8.1 MB)