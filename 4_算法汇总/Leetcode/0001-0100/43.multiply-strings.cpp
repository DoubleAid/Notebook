/*
 * @lc app=leetcode id=43 lang=cpp
 *
 * [43] Multiply Strings
 *
 * https://leetcode.com/problems/multiply-strings/description/
 *
 * algorithms
 * Medium (40.33%)
 * Likes:    6989
 * Dislikes: 3312
 * Total Accepted:    798.5K
 * Total Submissions: 2M
 * Testcase Example:  '"2"\n"3"'
 *
 * Given two non-negative integers num1 and num2 represented as strings, return
 * the product of num1 and num2, also represented as a string.
 * 
 * Note: You must not use any built-in BigInteger library or convert the inputs
 * to integer directly.
 * 
 * 
 * Example 1:
 * Input: num1 = "2", num2 = "3"
 * Output: "6"
 * Example 2:
 * Input: num1 = "123", num2 = "456"
 * Output: "56088"
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= num1.length, num2.length <= 200
 * num1 and num2 consist of digits only.
 * Both num1 and num2 do not contain any leading zero, except the number 0
 * itself.
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string multiply(string num1, string num2) {
        int n1 = num1.size(), n2 = num2.size();
    vector<int> result(n1 + n2, 0);

    for (int i = n1 - 1; i >= 0; i--) {
        for (int j = n2 - 1; j >= 0; j--) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int p1 = i + j, p2 = i + j + 1;
            int sum = mul + result[p2];

            result[p2] = sum % 10;
            result[p1] += sum / 10;
        }
    }

    string product;
    for (int num : result) {
        if (!(product.empty() && num == 0)) {  // Skip leading zeros
            product.push_back(num + '0');
        }
    }

    return product.empty() ? "0" : product;
    }
};
// @lc code=end

// 311/311 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 55.87 % of cpp submissions (8.4 MB)