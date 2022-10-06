这一张将介绍如何将本地打包好的 docker image 上传到 docker registry 上

### <font color="deepskyblue">docker run 的缺陷</font>
在使用 `docker run` 指令运行 docker container时，需要在本地手动部署， 这种方法不适用于实际的产品环境中， 所以需要 Kubernetes 这样的管理工具对 container 进行部署和管理。 也需要有一个地方能够让 Kubernetes 随时可以存取这些 docker image， 也就是 docker registry

### <font color="deepskyblue">什么是docker registry</font>
docker registry 是一个仓库， 用于存放各种 docker image， 仓库可以使公开的，也可以是私有的，只允许特定的人进行存取
官网传送门 --> [门](https://hub.docker.com/)

登录 docker
```
docker login
```
在上传 docker image 之前， 必须先建立一个 docker registry， 登陆后在 docker hub 点击 create->create repository
### <font color="deepskyblue">上传 docker image</font>
在 终端 输入 `docker image ls` 查看建立的image
使用 `docker tag` 将新建的 docker image 标上 tag
```
docker tag {conatainer id} {registry name}:{version ex: v1.0.0}
docker tag 59f3e3615488 zxcvbnius/docker-demo:v1.0.0
```
最后使用 `docker push`指令， 将标记好tag的image，上传到指定的registry
```
docker push {registry name}
```

### <font color="deepskyblue"></font>

[参考链接](https://ithelp.ithome.com.tw/articles/10192824)