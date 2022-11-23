![](./extra_images/v2-1846948dbfbe33d410644c80b614cd50_1440w.jpg)

KMP算法 （全称 Knuth-Morris-Pratt 字符串查找算法， 由三维发明者的姓氏命名）是可以在文本串s中快速查找模式串p的一种算法。

对于普通的暴力匹配， 就是逐个字符逐个字符的进行匹配， 如果当前字符串匹配成功（s[i] == p[j]）,就匹配下一个字符（i++, j++）,如果不适配， i 会进行回溯， j置为0 （i=i-j+1, j=0）, 代码如下：
```C++
int i=0, j=0;
while (i < s.length()) {
  if (s[i] == p[j]) {
    i++, j++;
  } else {
    i = i - j + 1, j = 0;
  }
  if (j == p.length()) {
    cout << i - j << endl;
    i = i - j + 1;
    j = 0;
  }
}
```