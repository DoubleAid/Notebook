# 192 [M] Word Frequency
Write a bash script to calculate the frequency of each word in a text file words.txt.
For simplicity sake, you may assume:
+ words.txt contains only lowercase characters and space ' ' characters.
+ Each word must consist of lowercase characters only.
+ Words are separated by one or more whitespace characters.

#### Example:
Assume that words.txt has the following content:
```
the day is sunny the the
the sunny is is
```
Your script should output the following, sorted by descending frequency:
```
the 4
is 3
sunny 2
day 1
```
**Note:**
+ Don't worry about handling ties, it is guaranteed that each word's frequency count is unique.
+ Could you write it in one-line using Unix pipes?

```bash
cat words.txt | (tr ' ' '\n' ) | tr -s '\n' | sort | uniq -c | sort -hr | awk -c '{print $2" "$1}'
```
命令行详解
+ `tr [option] [字符集1] [字符集2]`
    ```
    tr 命令用于转换或者删除文件中的字符
    参数说明:
        -c: 反选设定字符， 除了符合字符集1的部分不作处理， 剩余都进行转换
        -d: 删除指令字符
        -s: 缩减连续重复的字符成指定的单个字符
        -t: 缩减 SET1 的指定范围， 使之与 SET2 设定的长度相等

    实例说明:
        cat test.txt | tr a-z A-Z  将文件中的小写字母替换成 大写字母
    ```
+ sort
    ```

    ```
+ uniq
    ```
    
    ```
+ awk
  参考 awk 文档