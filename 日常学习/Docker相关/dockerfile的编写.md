<!-- TOC -->

- [**FROM** -> 他的妈妈是谁 (基础镜像是谁)](#from---他的妈妈是谁-基础镜像是谁)
- [**MAINTAINER** -> 告诉别人，你创造了它 （维护者的信息）](#maintainer---告诉别人你创造了它-维护者的信息)
- [**RUN** -> 你想让它干啥 （在命令前面加上 RUN）](#run---你想让它干啥-在命令前面加上-run)
- [**ADD** -> 往镜像里面放点文件 （COPY文件， 会自动解压）](#add---往镜像里面放点文件-copy文件-会自动解压)
  - [<font color="skyblue">COPY 命令</font>](#font-colorskybluecopy-命令font)
  - [<font color="skyblue">ADD 命令</font>](#font-colorskyblueadd-命令font)
- [**WORKDIR** -> 我是cd， 我今天化了妆， （进入当前工作目录）](#workdir---我是cd-我今天化了妆-进入当前工作目录)
- [**VOLUME** -> 给我一个放行李的地方， （目录挂载）](#volume---给我一个放行李的地方-目录挂载)
- [**EXPOSE** -> 我要打开的门是啥 （开放端口）](#expose---我要打开的门是啥-开放端口)
- [**RUN** -> 运行进程](#run---运行进程)

<!-- /TOC -->

## **FROM** -> 他的妈妈是谁 (基础镜像是谁)

## **MAINTAINER** -> 告诉别人，你创造了它 （维护者的信息）

## **RUN** -> 你想让它干啥 （在命令前面加上 RUN）
镜像的操作命令 当我们需要定制一个镜像的时候，肯定是要运行一些命令（安装一些依赖，修改一些配置等等）来对基础镜像做一些修改的，一般使用RUN 具体命令来操作，RUN的默认权限是sudo。 需要注意的是，如果你需要执行多个RUN操作，那最好把它们合并在一行 (用&&连接)，因为每执行一次RUN就会在docker上新建一层镜像，所以分开来写很多个RUN的结果就是会导致整个镜像无意义的过大膨胀。 正确的做法: bash RUN apt-get update && apt-get install vim

容器启动时执行的命令 需要用CMD来执行一些容器启动时的命令，注意与RUN的区别，CMD是在docker run执行的时候使用，而RUN则是在docker build的时候使用，还有，划重点，一个Dockerfile只有最后一个CMD会起作用。 栗子： bash CMD ["/usr/bin/wc","--help"]

## **ADD** -> 往镜像里面放点文件 （COPY文件， 会自动解压）
### <font color="skyblue">COPY 命令</font>
copy 命令的格式 `COPY <src> <dest>`  
COPY 命令区别于 ADD 命令的一个用法是在 multistage 场景下, 在 multistage 的用法中，可以使用 COPY 命令把前一阶段构建的产物拷贝到另一个镜像中  
`COPY --from=0 /go/src/github.com/sparkdevo/href-counter/app .`

### <font color="skyblue">ADD 命令</font>
除了不能用在 multistage 的场景下，ADD 命令可以完成 COPY 命令的所有功能，并且还可以完成两类超酷的功能：
+ 解压压缩文件并把它们添加到镜像中
  ```dockerfile
  WORKDIR /app
  ADD nickdir.tar.gz .
  ```
+ 从 url 拷贝文件到镜像中
  docker 官方建议我们当需要从远程复制文件时，最好使用 curl 或 wget 命令来代替 ADD 命令。原因是，当使用 ADD 命令时，会创建更多的镜像层，当然镜像的 size 也会更大(下面的两段代码来自 docker 官方文档)：
  ```dockerfile
  ADD http://example.com/big.tar.xz /usr/src/things/
  RUN tar -xJf /usr/src/things/big.tar.xz -C /usr/src/things
  RUN make -C /usr/src/things all
  ```
  如果使用下面的命令，不仅镜像的层数减少，而且镜像中也不包含 big.tar.xz 文件：
  ```dockerfile
  RUN mkdir -p /usr/src/things \
      && curl -SL http://example.com/big.tar.xz \
      | tar -xJC /usr/src/things \
      && make -C /usr/src/things all
  ```

## **WORKDIR** -> 我是cd， 我今天化了妆， （进入当前工作目录）

## **VOLUME** -> 给我一个放行李的地方， （目录挂载）

## **EXPOSE** -> 我要打开的门是啥 （开放端口）

## **RUN** -> 运行进程
























