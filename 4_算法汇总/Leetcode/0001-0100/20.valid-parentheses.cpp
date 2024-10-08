/*
 * @lc app=leetcode id=20 lang=cpp
 *
 * [20] Valid Parentheses
 */

// @lc code=start
class Solution {
public:
    bool isValid(string s) {
        unordered_map<char, int> hash_map = {
            {'[', -1},
            {']', 1},
            {'(', -2},
            {')', 2},
            {'{', -3},
            {'}', 3}
        };
        vector<int> record;
        for (int i = 0; i < s.size(); i++) {
            if (hash_map[s[i]] < 0) {
                record.emplace_back(hash_map[s[i]]);
            } else {
                if (record.empty()) return false;
                if (record.back() + hash_map[s[i]] == 0) { 
                    record.pop_back();
                } else {
                    return false;
                }

            }
        }
        if (record.empty()) return true;
        return false;
    }
};
// @lc code=end

// 98/98 cases passed (4 ms)
// Your runtime beats 13.87 % of cpp submissions
// Your memory usage beats 6.63 % of cpp submissions (8.8 MB)