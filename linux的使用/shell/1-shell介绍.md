一个常见的shell脚本例如：
```
#!/bin/bash
echo "Hello World!"
```
"#!" 是一个约定的标记，它告诉系统这个脚本需要什么解释器来执行，即使用哪一种Shell。echo命令用于向窗口输出文本。

### 运行shell脚本
```
chmod +x ./test.sh #使脚本具有可执行权限
./test.sh #执行脚本
```

### 变量的操作
```shell
url = "hello"
echo $url

# 可以直接修改
url = "gene"

# 赋值
link = ${url}

# 可以设置为只读访问
url = "ted"
readonly url

# 删除变量
unset url
```

参考链接：http://c.biancheng.net/cpp/view/7005.html
