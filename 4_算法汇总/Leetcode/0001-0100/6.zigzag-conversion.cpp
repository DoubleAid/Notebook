/*
 * @lc app=leetcode id=6 lang=cpp
 *
 * [6] Zigzag Conversion
 */

// @lc code=start
class Solution {
public:
    string convert(string s, int numRows) {
        if (numRows == 1) return s;
        string res;
        for (int i = 0; i < numRows; i++) {
            int start = i;
            while (start < s.size()) {
                res += s[start];
                if (i != 0 && i != numRows-1) {
                    int bias = start + 2*(numRows-1-i);
                    if (bias < s.size()) {
                        res += s[bias];
                    }
                }
                start += 2*(numRows - 1);
            }
        }
        return res;
    }
};
// @lc code=end

// 数学推理
// 0                 2n-2
// 1            2n-3 2n-1
// 2       2n-4
// .     .
// .   n
// n-1

// 1157/1157 cases passed (3 ms)
// Your runtime beats 97.75 % of cpp submissions
// Your memory usage beats 91.77 % of cpp submissions (10.3 MB)