# 0003 Longest Substring Without Repeating Characters
Given a string s, find the length of the longest substring without repeating characters.

#### Example 1:
```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```
#### Example 2:
```
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```
#### Example 3:
```
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

#### Constraints:
+ 0 <= s.length <= 5 * 104
+ s consists of English letters, digits, symbols and spaces.

#### 思路
用一个 map 记录字符出现的最后位置

#### C++
```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> index_;
        int max_len = 0, start = 0, end = 0;
        for(; end < s.length(); end++) {
            if(index_.count(s[end]) == 0) {
                index_.emplace(s[end], end);
            } else {
                max_len = max(max_len, end - start);
                while(start < index_[s[end]]) {
                    index_.erase(s[start++]);
                }
                start++;
                index_[s[end]] = end;
            }
        }
        return max(max_len, end-start);
    }
};
```