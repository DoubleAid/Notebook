/*
 * @lc app=leetcode id=9 lang=cpp
 *
 * [9] Palindrome Number
 */

// @lc code=start
class Solution {
public:
    // bool isPalindrome(int x) {
    //     if (x < 0) return false;
    //     vector<int> val;
    //     while (x != 0) {
    //         val.emplace_back(x%10);
    //         x /= 10;
    //     }
    //     int vlen = val.size();
    //     for (int i = 0; i < vlen / 2; i++) {
    //         if (val[i] != val[vlen - i - 1]) return false;
    //     }
    //     return true;
    // }
    // 效率比较低，
    // 11511/11511 cases passed (19 ms)
    // Your runtime beats 8.87 % of cpp submissions
    // Your memory usage beats 5.32 % of cpp submissions (11.8 MB)

    bool isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;

        int right_part = 0;
        while (x > right_part) {
            right_part = right_part * 10 + x % 10;
            x /= 10;
        }
        // 121 -> 1 12
        return x == right_part || x == right_part / 10;
    }
};
// @lc code=end

// 11511/11511 cases passed (8 ms)
// Your runtime beats 60.21 % of cpp submissions
// Your memory usage beats 48.46 % of cpp submissions (8.3 MB)


