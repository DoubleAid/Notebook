so 文件找不到通常分为两种，
+ 一种是这个lib库你没有，或者下载的源文件没有编译。
+ 另一种是 这个 so文件 编译时没有被链接到

通过 `sudo find . -name "libxxx.so"`确定该 文件是否存在

### 第一种就是直接下载或编译相应的so文件

### 第二种需要将相应的so文件添加到本地的配置链接之中，
第一种试讲该文件拷贝到 `usr/local/lib 内`
在 /etc/ld.so.conf.d 中新建一个 configure 文件，写入 so文件所在的位置
执行`ldconfig`命令更新 /etc/ld.so.cache 文件内容