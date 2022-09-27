- [docker-compose](#docker-compose)
    - [yml 配置参考](#yml-配置参考)
        - [version](#version)
        - [build](#build)

[参考文档](https://www.runoob.com/docker/docker-compose.html)

# Docker Compose
Compose 是用于定义和运行多容器 Docker 应用程序的工具，通过 Compose 使用 YML 文件来配置 应用程序需要的所有服务，然后使用一个命令就可以从 YML 文件配置中创建并启动所有服务

compose 的使用分为三个步骤
+ 使用 Dockerfile 定义应用程序的环境
+ 使用 docker-compose.yml 定义构成应用程序的服务， 这样他们可以在隔离环境中一起运行
+ 最后 实行 `docker-compose up` 命令来启动并运行整个应用程序

## yml 配置参考
#### version
指定本 yml 依照 compose 的 哪个版本制定

#### build

## 启动并运行整个应用程序
```
# 在根目录下 执行以下命令启动应用程序
docker-compose up

# 如果想在后台执行该服务 可以加上 -d 参数
docker-compose up -d

# 参考命令
docker-compose -f docker-compose.yml up -d ros_core map_db_node map_db_update_node
```