/*
 * @lc app=leetcode id=38 lang=cpp
 *
 * [38] Count and Say
 */
#include <string>

using namespace std;

// @lc code=start
class Solution {
public:
    string countAndSay(int n) {
        string result = "1";
        int count = 1;
        while (count < n) {
            result = generateNewString(result);
            count++;
        }
        return result;
    }

    string generateNewString(string s) {
        string result;
        int index = 0;
        while (index < s.size()) {
            int count = 1;
            while (index+1 < s.size() && s[index+1] == s[index]) {
                index++;
                count++;
            }
            result += (std::to_string(count) + s[index]);
            index++;
        }
        return result;
    }
};
// @lc code=end

// 30/30 cases passed (3 ms)
// Your runtime beats 70.5 % of cpp submissions
// Your memory usage beats 27.57 % of cpp submissions (9.3 MB)