/*
 * @lc app=leetcode id=17 lang=cpp
 *
 * [17] Letter Combinations of a Phone Number
 */
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        unordered_map<int, vector<string>> hash_map = {
            {2, vector<string>{"a", "b", "c"}},
            {3, vector<string>{"d", "e", "f"}},
            {4, vector<string>{"g", "h", "i"}},
            {5, vector<string>{"j", "k", "l"}},
            {6, vector<string>{"m", "n", "o"}},
            {7, vector<string>{"p", "q", "r", "s"}},
            {8, vector<string>{"t", "u", "v"}},
            {9, vector<string>{"w", "x", "y", "z"}},           
        };
        for (int i = 0; i < digits.size(); i++) {
            int val = digits[i] - '0';
            if (res.empty()) {
                res = hash_map[val];
                continue;
            }
            vector<string> new_res;
            for (int j = 0; j < res.size(); j++) {
                for (int k = 0; k < hash_map[val].size(); k++) {
                    new_res.emplace_back(res[j] + hash_map[val][k]);
                }
            }
            res = new_res;
        }
        return res;
    }
};
// @lc code=end

// 25/25 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 20.96 % of cpp submissions (8.4 MB)