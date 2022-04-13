这一章将介绍如何将程序打包成 Docker Image，建造一个属于你自己的container，并能够在 Kubernetes 上运行
+ 安装 Docker Container
+ 将程序打包成 Docker Image
+ 在本地上运行 Containerized App

## 安装 Docker Engine
在开始打造 Docker Container 之前， 需要先安装 Docker Engine --> [docker 官网传送门](https://docs.docker.com/engine/install/)

ubuntu
```
sudo apt-get update & sudo apt-get install docker.io
```

## 将程序 封装成 docker image
接下来将以一个 Nodejs app 为例， 将该程序打包成 Docker Image，下面介绍即将用到的三个文件
### <font color="deepskyblue">Dockerfile</font>
在将程序docker化时，需要一个专属于该程序的Dockerfile
以 Nodejs App为例
```python
FROM node:6.2.2 # 载入程序需要的执行环境， 会根据不同的需求下载不同的映像， 这里指 node v6.2.2
WORKDIR /app # docker 中的linux系统会创建一个目录 /app
ADD . /app # 将与 Dockerfile 同一目录下的所有文件添加到 Linux 的 /app 目录下
RUN npm install # 运行 npm install， npm install 将会下载 nodejs 依赖的 lib
EXPOSE 300 # 设置container的id， 方便与外界沟通
CMD npm start 最后通过 npm start 运行 Nodejs App
```
其他命令可以参考 [《Docker 从入门到入土》](https://philipzheng.gitbook.io/docker_practice/dockerfile/instructions)

### <font color="deepskyblue">index.js</font>
通过 express 套件搭建小型的 nodejs api server