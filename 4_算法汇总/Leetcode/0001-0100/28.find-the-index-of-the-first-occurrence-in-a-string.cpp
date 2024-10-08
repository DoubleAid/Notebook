/*
 * @lc app=leetcode id=28 lang=cpp
 *
 * [28] Find the Index of the First Occurrence in a String
 */

// @lc code=start
class Solution {
public:
    vector<int> kmpProcess(string needle) {
    int n = needle.length();
    vector<int> lps(n, 0);
    for (int i = 1, len = 0; i < n;) {
        if (needle[i] == needle[len]) {
            lps[i++] = ++len;
        } else if (len != 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;
    int n = haystack.length(), m = needle.length();
    vector<int> lps = kmpProcess(needle);
    for (int i = 0, j = 0; i < n;) {
        if (haystack[i] == needle[j]) {
            i++, j++;
            if (j == m) {
                return i - m; // Match found
            }
        } else if (j > 0) {
            j = lps[j - 1];
        } else {
            i++;
        }
    }
    return -1; // No match found
}
};
// @lc code=end


// KMP 算法