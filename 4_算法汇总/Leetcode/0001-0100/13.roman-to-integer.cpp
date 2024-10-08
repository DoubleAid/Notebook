/*
 * @lc app=leetcode id=13 lang=cpp
 *
 * [13] Roman to Integer
 */
#include <unordered_map>
#include <string>
using namespace std;
// @lc code=start
class Solution {
public:
    int romanToInt(string s) {
        int res = 0;
        int p = 0;
        unordered_map<char, int> hash_map = {
            {'M', 1000},
            {'D', 500},
            {'C', 100},
            {'L', 50},
            {'X', 10},
            {'V', 5},
            {'I', 1},
        };
        while (p < s.size()) {
            while (p < s.size() - 1 && hash_map[s[p]] >= hash_map[s[p+1]]) {
                res += hash_map[s[p]];
                p++;
            }

            // if (p == s.size() - 1) return res + hash_map[s[p]];

            if (p < s.size() - 1 && hash_map[s[p]] < hash_map[s[p+1]]) {
                res += (hash_map[s[p+1]] - hash_map[s[p]]);
                p += 2;
            } else {
                res += hash_map[s[p]];
                p++;
            }
        } 
        return res;
    }
};
// @lc code=end

// 3999/3999 cases passed (8 ms)
// Your runtime beats 65.02 % of cpp submissions
// Your memory usage beats 57.22 % of cpp submissions (12.7 MB)