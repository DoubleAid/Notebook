g3logger 安装时未指定dversion， 编译出来的 cmake 文件缺失，需要指定相应的 dversion

```
git clone https://github.com/KjellKod/g3log.git \
    && cd g3log \
    && git checkout 376c417ad170228fa6d1b9994a6e07a2ac143a51 \
    && mkdir build \
    && cd build \
    && cmake -DVERSION=1.3.2-78 .. \
    && make \
    && make install \
    && make clean
```
