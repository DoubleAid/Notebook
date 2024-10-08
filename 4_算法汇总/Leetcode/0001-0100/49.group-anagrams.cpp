/*
 * @lc app=leetcode id=49 lang=cpp
 *
 * [49] Group Anagrams
 *
 * https://leetcode.com/problems/group-anagrams/description/
 *
 * algorithms
 * Medium (68.48%)
 * Likes:    18977
 * Dislikes: 608
 * Total Accepted:    2.9M
 * Total Submissions: 4.2M
 * Testcase Example:  '["eat","tea","tan","ate","nat","bat"]'
 *
 * Given an array of strings strs, group the anagrams together. You can return
 * the answer in any order.
 * 
 * An Anagram is a word or phrase formed by rearranging the letters of a
 * different word or phrase, typically using all the original letters exactly
 * once.
 * 
 * 
 * Example 1:
 * Input: strs = ["eat","tea","tan","ate","nat","bat"]
 * Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
 * Example 2:
 * Input: strs = [""]
 * Output: [[""]]
 * Example 3:
 * Input: strs = ["a"]
 * Output: [["a"]]
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= strs.length <= 10^4
 * 0 <= strs[i].length <= 100
 * strs[i] consists of lowercase English letters.
 * 
 * 
 */
#include <vector>
#include <algorithm>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        if (strs.empty()) return vector<vector<string>>{{}};
        vector<string> new_group{strs[0]};
        string seq = sort()
        int 
    }
};
// @lc code=end

