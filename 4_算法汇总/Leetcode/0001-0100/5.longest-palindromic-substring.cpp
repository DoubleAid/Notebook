/*
 * @lc app=leetcode id=5 lang=cpp
 *
 * [5] Longest Palindromic Substring
 */
#include <iostream>
#include <string>
#include <vector>
using namespace std;


// @lc code=start
class Solution {
public:
 string longestPalindrome(string s) {
        int slen = s.size();
        if (slen == 0) return "";
        
        // 预处理字符串，将普通字符串转换为带有间隔符的形式，以统一奇偶长度回文的处理逻辑
        string t = "#";
        for (char c : s) {
            t += c;
            t += '#';
        }

        int newLen = t.size();
        vector<int> p(newLen, 0);
        int C = 0, R = 0;  // 中心和右边界
        int maxLen = 0, centerIndex = 0;

        for (int i = 0; i < newLen; i++) {
            // 利用对称性初始化半径
            int mirror = 2 * C - i;
            if (i < R) {
                p[i] = min(R - i, p[mirror]);
            }

            // 中心扩展
            int left = i - (1 + p[i]);
            int right = i + (1 + p[i]);
            while (left >= 0 && right < newLen && t[left] == t[right]) {
                p[i]++;
                left--;
                right++;
            }

            // 更新中心和右边界
            if (i + p[i] > R) {
                C = i;
                R = i + p[i];
            }

            // 更新最大回文子串信息
            if (p[i] > maxLen) {
                maxLen = p[i];
                centerIndex = i;
            }
        }

        // 从处理后的字符串中提取最大回文子串，并去除间隔符
        int start = (centerIndex - maxLen) / 2;  // 计算原始字符串中的起始位置
        return s.substr(start, maxLen);
    }
};
// @lc code=end

int main() {
    Solution s = Solution();
    string m = "babad";
    cout << s.longestPalindrome(m) << endl;
    return 0;
}


// 142/142 cases passed (4 ms)
// Your runtime beats 98.08 % of cpp submissions
// Your memory usage beats 66.21 % of cpp submissions (9.8 MB)