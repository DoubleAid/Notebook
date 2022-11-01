# awk
awk 是处理文本文件的一个应用程序， 几乎所有linux系统都自带这个程序

它依次处理文件的每一行， 并读取里面的每一个字段。 对于 日志、CSV 那样的每行格式相同的文本文件， awk 是最方便的工具
## 基本用法
```bash
awk '{print $0}' demo.txt
```
`demo.txt` 就是 `awk` 要处理的文件。 前面单引号内部有一个大括号， 里面的内容 就是对每一行要做的处理动作 `print $0`, `print` 是打印命令， `$0` 代表当前行， 因此上面命令运行的结果就是把每一行都打印出来

`awk` 会根据空格和制表符， 将每一行分成若干字段， 依次用 `$1`, `$2`, `$3` 来代表第一个字段， 第二个字段， 第三个字段等等
```bash
echo 'this is a test' | awk '{print $3}'

# 输出为 a
```
也可以自定义分割符
```bash
echo "root:x:0:0:root:/root:/usr/bin/zsh" | awk -F ':' '{ print $1 }'
# 输出为 root

# 也可以指定多个分割符
echo "There are orange,apple,mongo" | awk -F '[ ,]' '{print $1,$3,$4}'
# 输出为 There orange apple
```

除了 `$ + 数字` 表示某个字段， `awk` 还提供其他一些变量。

变量 `NF` 表示当前行有多少个字段， 因此 `$NF` 就代表最后一个字段。
```bash
echo "this is a test" | awk '{print $NF}'
# 输出为 test

echo "this is a test" | awk '{print $(NF-1)}'
# 输出为 a
```

上面所有的例子里 `print` 命令后面的逗号， 表示输出的时候， 两个部分之间使用空格分割

### 内置变量
`NR` 表示当前处理的是第几行

```bash
awk -F ':' '{print NR ") " $1}' demo.txt
# 输出如下
1） root
2） daemon
3） bin
```

`awk` 其他的内置变量如下
+ FILENAME: 当前文件名
+ FS: 字段分隔符， 默认是空格和制表符
+ RS： 行分隔符， 默认是换行符
+ OFS: 输出字段的分隔符， 用于打印时分割字段， 默认是空格
+ ORS: 输出记录的分隔符， 用于打印时分割记录， 默认是换行符
+ OFMT： 数字输出格式， 默认是 %.6g

### 内置函数
`awk` 还提供了一些内置函数， 方便对原始数据的处理
函数 `toupper()` 用于将字符转为大写
```bash
echo "hello world to you" | awk '{ print toupper($1) }'
# 输出为 HELLO
```
其他常见的函数如下：
+ tolower(): 转为小写
+ length(): 字符串长度
+ substr(): 返回子字符串
+ sin(): 正弦
+ cos(): 余弦
+ sqrt(): 平方根
+ rand(): 随机数

### 条件
`awk` 允许指定输出条件， 只输出符合条件的行
```bash
# print 之前是一个正则表达式， 只输出包含 `usr` 的行
awk '/usr/ {print $1}' demo.txt
# 输出 奇数行
awk 'NR % 2 == 1 {print $1}' demo.txt
# 输出第三行以后的行
awk 'NR > 3 {print $1}' demo.txt
# 输出第一个字段等于指定值的行
awk '$1 == "root" {print $1}' demo.txt
awk '$1 == "root" || $1 == "bin" {print $1}' demo.txt
# if 语句
awk '{if ($1 > "m") print $1}' demo.txt
awk '{if ($1 > "m") print $1; else print $2}'
```
## 使用参考
+ 获取命令行窗口的高宽， 打印输出宽度

参考文档
+ https://www.ruanyifeng.com/blog/2018/11/awk.html