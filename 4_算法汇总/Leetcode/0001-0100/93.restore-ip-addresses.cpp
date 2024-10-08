/*
 * @lc app=leetcode id=93 lang=cpp
 *
 * [93] Restore IP Addresses
 */

#include <string>
#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<string> restoreIpAddresses(string s) {
        return splitIPAddress(s, 3);
    }

    vector<string> splitIPAddress(string s, int part) {
        vector<string> result;
        if (part == 0 && isValid(s)) return vector<string>{s};
        for (int i = 1; i <=3 && i < s.size(); i++) {
            string cur = s.substr(0, i);
            if (isValid(cur)) {
                vector<string> temp = splitIPAddress(s.substr(i), part-1);
                for (string ans: temp) {
                    result.emplace_back(cur + "." + ans);
                }
            }
        }
        return result;
    }

    bool isValid(string s) {
        if (s.size() == 0 || (s[0] == '0' && s.size() > 1) || s.size() > 3 ||(stoi(s) > 255)) return false;
        return true;
    }
};
// @lc code=end

// 146/146 cases passed (5 ms)
// Your runtime beats 37.65 % of cpp submissions
// Your memory usage beats 21.73 % of cpp submissions (11.4 MB)