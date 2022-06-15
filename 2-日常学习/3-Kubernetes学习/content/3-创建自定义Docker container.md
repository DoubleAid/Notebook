这一章将介绍如何将程序打包成 Docker Image，建造一个属于你自己的container，并能够在 Kubernetes 上运行
+ 安装 Docker Container
+ 将程序打包成 Docker Image
+ 在本地上运行 Containerized App

## <font color="coral">安装 Docker Engine</font>
在开始打造 Docker Container 之前， 需要先安装 Docker Engine --> [docker 官网传送门](https://docs.docker.com/engine/install/)

ubuntu
```
sudo apt-get update & sudo apt-get install docker.io
```

## <font color="coral">将程序 封装成 docker image</font>
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
通过 express 套件搭建小型的 nodejs api server, server 运行在 port 3000 上， 并且有一个 endpoint 会返回 "hello world"字符串
index.js
```js
var express = require('express');
var app = express();
app.get('/', function(req, res) {
  res.send('Hello World!');
});
var server = app.listen(3000, function() {
  var host = server.address().address;
  var port = server.address().port;
  
  console.log("Example app listening at 'http://%s:%s'", host, port);
})
```
### <font color="deepskyblue">package.json</font>
package.json 会记录运行程序所需要的依赖
package.json
```json
{
  "name": "myapp",
  "version": "1.0.0",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.16.2"
  }
}
```

### <font color="deepskyblue">Docker build</font>
新建文件夹 `docker-demo` 保存上述的三个文件
```
(base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Code/docker-demo$ ls -l
total 12
-rw-r--r-- 1 guanggang.bian domain users  81 4月  14 10:29 Dockerfile
-rw-r--r-- 1 guanggang.bian domain users 315 4月  14 10:30 index.js
-rw-r--r-- 1 guanggang.bian domain users 143 4月  14 10:30 package.json
```

在该文件夹下执行 build 命令
```
docker build .
...此处省略...
Removing intermediate container e05fca8d7dc2
 ---> 7cc64731ad1f
Step 5/6 : EXPOSE 300
 ---> Running in 31c4558da51d
Removing intermediate container 31c4558da51d
 ---> 3c564571fa21
Step 6/6 : CMD npm start
 ---> Running in 3e9c20ac1eff
Removing intermediate container 3e9c20ac1eff
 ---> ca63b53ee571
Successfully built ca63b53ee571
```
当出现 `Successfully built ca63b53ee571` 表示 docker image
接着可以输入 `docker image ls` 指令查看刚刚建立好的 image
```
docker image ls
REPOSITORY                                                         TAG          IMAGE ID       CREATED          SIZE
<none>                                                             <none>       ca63b53ee571   10 minutes ago   665MB
<none>                                                             <none>       f338e5ca8f1b   4 months ago     22.7GB
<none>                                                             <none>       24e1021c050e   4 months ago     22.7GB
```

## <font color="coral">在本地运行 containerized app</font>
在打包好 Docker Image 之后， 就可以通过 docker指令 运行 docker container， 复制刚刚的 docker image ID， 输入指令
```
$ sudo docker run -p 3000:3000 -it ca63b53ee571
[sudo] password for guanggang.bian: 
npm info it worked if it ends with ok
npm info using npm@3.9.5
npm info using node@v6.2.2
npm info lifecycle myapp@1.0.0~prestart: myapp@1.0.0
npm info lifecycle myapp@1.0.0~start: myapp@1.0.0

> myapp@1.0.0 start /app
> node index.js

Example app listening at 'http://:::3000'
```

看到 `Example app listening at 'http://:::3000'` 表示 docker image 已经跑起来了， 可以通过访问 `http://127.0.0.1:3000/`