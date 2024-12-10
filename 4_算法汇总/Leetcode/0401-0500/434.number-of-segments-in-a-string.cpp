/*
 * @lc app=leetcode id=434 lang=cpp
 *
 * [434] Number of Segments in a String
 *
 * https://leetcode.com/problems/number-of-segments-in-a-string/description/
 *
 * algorithms
 * Easy (36.39%)
 * Likes:    800
 * Dislikes: 1272
 * Total Accepted:    200.2K
 * Total Submissions: 550.9K
 * Testcase Example:  '"Hello, my name is John"'
 *
 * Given a string s, return the number of segments in the string.
 * 
 * A segment is defined to be a contiguous sequence of non-space characters.
 * 
 * 
 * Example 1:
 * 
 * Input: s = "Hello, my name is John"
 * Output: 5
 * Explanation: The five segments are ["Hello,", "my", "name", "is", "John"]
 * 
 * 
 * Example 2:
 * 
 * Input: s = "Hello"
 * Output: 1
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s.length <= 300
 * s consists of lowercase and uppercase English letters, digits, or one of the
 * following characters "!@#$%^&*()_+-=',.:".
 * The only space character in s is ' '.
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int countSegments(string s) {
        int segment_count = 0;
        int index = 0;
        while (index < s.size() && s[index] == ' ') index++;
        if (index == s.size()) return 0;
        while (index < s.size()) {
            if (s[index] == ' ' && s[index-1] != ' ') segment_count++;
            index++; 
        }
        if (s[index-1] != ' ') segment_count++;
        return segment_count;
    }
};
// @lc code=end

