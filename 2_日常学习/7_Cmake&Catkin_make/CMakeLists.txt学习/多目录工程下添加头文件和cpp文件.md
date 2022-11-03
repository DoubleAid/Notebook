参考链接 https://blog.csdn.net/ktigerhero3/article/details/70313350


使用 `aux_source_directory(path/to/dir Name)` 添加 `path/to/dir` 下的 cpp 文件， 
在 add_exectuable 中添加 Name 变量， 例如
```
add_executable(app
               ${Name}
)
```