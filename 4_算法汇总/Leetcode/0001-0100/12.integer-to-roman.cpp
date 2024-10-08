/*
 * @lc app=leetcode id=12 lang=cpp
 *
 * [12] Integer to Roman
 */

#include <string>

using namespace std;

// @lc code=start
class Solution {
public:
    string intToRoman(int num) {
        string res;
        int num_1000 = num / 1000;
        num = num % 1000;
        for (int i = 0; i < num_1000; i++) {
            res += 'M';
        }
        if (num >= 900) {
            res += "CM";
            num -= 900;
        } else if (num >= 500) {
            res += 'D';
            num -= 500;
        } else if (num >= 400) {
            res += "CD";
            num -= 400;
        }
        int num_100 = num / 100;
        num = num % 100;
        for (int i = 0; i < num_100; i++) {
            res += 'C';
        }
        if (num >= 90) {
            res += "XC";
            num -= 90;
        } else if (num >= 50) {
            res += 'L';
            num -= 50;
        } else if (num >= 40) {
            res += "XL";
            num -= 40;
        }
        int num_10 = num / 10;
        num = num % 10;
        for (int i = 0; i < num_10; i++) {
            res += 'X';
        }
        if (num == 9) {
            res += "IX";
            num -= 9;
        } else if (num >= 5) {
            res += 'V';
            num -= 5;
        } else if (num == 4) {
            res += "IV";
            num -= 4;
        }
        for (int i = 0; i < num; i++) {
            res += 'I';
        }
        return res;
    }
};
// @lc code=end

// 3999/3999 cases passed (3 ms)
// Your runtime beats 87.2 % of cpp submissions
// Your memory usage beats 99.59 % of cpp submissions (7.6 MB)
