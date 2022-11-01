# 709 To Lower Case 转化小写
Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.

```
Example 1:

Input: "Hello"
Output: "hello"
Example 2:

Input: "here"
Output: "here"
Example 3:

Input: "LOVELY"
Output: "lovely"
```
#### 思路
大写字母 为 65 -- 90
大小写转化 + 32
#### python
转ascii -> ord()
转字符 -> chr() 
```python
class Solution:
    def toLowerCase(self, str: str) -> str:
        lst = list(str)

        for i in range(len(lst)):
            if ord(lst[i]) >= 65 and ord(lst[i]) <= 90:
                lst[i] = chr(ord(lst[i]) + 32)
        return "".join(lst)
```