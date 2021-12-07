### conan 安装

首先要切换python版本至python3

```
pip3 install conan
```

如果显示命令不存在

```
sudo ln -s ~/.local/bin/conan /usr/bin/conan
```

常见命令

```
# 查看 远程列表
conan remote list

# 查找 某一个 软件
conan search xxx --remote conancenter

# 下载
conan install xxx/版本@
```