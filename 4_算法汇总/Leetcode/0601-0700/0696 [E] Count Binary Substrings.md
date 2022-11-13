# 696 [E] Count Binary Substrings
Given a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

#### Example 1:
```
Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
```
#### Example 2:
```
Input: s = "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
```

#### Constraints:
+ 1 <= s.length <= 105
+ s[i] is either '0' or '1'.

#### 思路和解题方法
这道题让我想到了最大回环子串那道题， 那道题里面选取中心点， 用右边的点和左边的点进行对比校验， 这道题没有那么麻烦， 但也用到了这个思路
时间复杂度为 $O(n)$
```cpp
class Solution {
public:
    int countBinarySubstrings(string s) {
        int count = 0;
        int i = 0;
        while ( i < s.length()) {
          int j = 0;
          while (i+j < s.length() && i - j > 0 && s[i+j] == s[i] && s[i-j-1] != s[i]) j++;
          if (j != 0) {
            i += j;
          }
          else {
            i++;
          }
          count += j;
        }
        return count;
    }
};
```

第二种方法是添加一个标记为记录当前左边的最长长度，右边只要交验这个长度内的子串就可以了， 这样可以在运算时减少一个判断， 对整体的速度没太大的影响