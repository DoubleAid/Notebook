/*
 * @lc app=leetcode id=405 lang=cpp
 *
 * [405] Convert a Number to Hexadecimal
 *
 * https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/
 *
 * algorithms
 * Easy (48.36%)
 * Likes:    1326
 * Dislikes: 221
 * Total Accepted:    157.4K
 * Total Submissions: 318.4K
 * Testcase Example:  '26'
 *
 * Given a 32-bit integer num, return a string representing its hexadecimal
 * representation. For negative integers, two’s complement method is used.
 * 
 * All the letters in the answer string should be lowercase characters, and
 * there should not be any leading zeros in the answer except for the zero
 * itself.
 * 
 * Note: You are not allowed to use any built-in library method to directly
 * solve this problem.
 * 
 * 
 * Example 1:
 * Input: num = 26
 * Output: "1a"
 * Example 2:
 * Input: num = -1
 * Output: "ffffffff"
 * 
 * 
 * Constraints:
 * 
 * 
 * -2^31 <= num <= 2^31 - 1
 * 
 * 
 */

#include <string>

using namespace std;

// @lc code=start
class Solution {
public:
    string toHex(int num) {
        int zero_num = 0;
        string prefix = "0123456789abcdef";
        string ret = "";
        for (int i = 0; i < 8; i++) {
            int tmp = 0xf & num;
            num = num >> 4;
            if (tmp == 0) {
                zero_num++;
                continue;
            } else {
                ret = string(zero_num, '0') + ret;
                zero_num = 0;
            }
            ret = prefix[tmp] + ret;
        }
        if (zero_num == 8) return "0";
        return ret;
    }
};
// @lc code=end

