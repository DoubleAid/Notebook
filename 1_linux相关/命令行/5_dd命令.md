Linux dd 命令用于读取、转换并输出数据。

dd 可从标准输入或文件中读取数据，根据指定的格式来转换数据，再输出到文件、设备或标准输出。

**参数说明:**

+ if=文件名：输入文件名，默认为标准输入。即指定源文件。
+ of=文件名：输出文件名，默认为标准输出。即指定目的文件。
+ ibs=bytes：一次读入bytes个字节，即指定一个块大小为bytes个字节。
+ obs=bytes：一次输出bytes个字节，即指定一个块大小为bytes个字节。
+ bs=bytes：同时设置读入/输出的块大小为bytes个字节。
+ cbs=bytes：一次转换bytes个字节，即指定转换缓冲区大小。
+ skip=blocks：从输入文件开头跳过blocks个块后再开始复制。
+ seek=blocks：从输出文件开头跳过blocks个块后再开始复制。
+ count=blocks：仅拷贝blocks个块，块大小等于ibs指定的字节数。
+ conv=<关键字>，关键字可以有以下11种：
+ conversion：用指定的参数转换文件。
+ ascii：转换ebcdic为ascii
+ ebcdic：转换ascii为ebcdic
+ ibm：转换ascii为alternate ebcdic
+ block：把每一行转换为长度为cbs，不足部分用空格填充
+ unblock：使每一行的长度都为cbs，不足部分用空格填充
+ lcase：把大写字符转换为小写字符
+ ucase：把小写字符转换为大写字符
+ swap：交换输入的每对字节
+ noerror：出错时不停止
+ notrunc：不截短输出文件
+ sync：将每个输入块填充到ibs个字节，不足部分用空（NUL）字符补齐。
+ --help：显示帮助信息
+ --version：显示版本信息