minikube 是由 google 发布的一款轻量级工具，可以让开发者在本地轻易的搭建一个 Kubernetes Cluster,快速上手 Kubernetes 的指令和环境。  
minikube 会在本地运行一个 virtual machine。 并在这个虚拟机里建立一个 single-node Kubernetes cluster, 本身并不支持 high availability， 也不推荐在实际应用上运行

## 安装 Minikube
Minikube 支持 windows，macos， linux， 安装过程：
+ 确认本机是否已经安装了virtualization software，例如 virtualbox
+ 手动安装 kubectl 命令行插件
+ 从 github 下载 minikube 套件
+ 在 minikube 上执行 hello-minikube app

### 确认本机是否已经安装了virtualization software
minikube 会在本机运行一个 vm（虚拟机），所以在开始安装 minikube 之前，需要先确认本机是否安装了 虚拟化软件 --> [virtualbox 传送门](https://www.virtualbox.org/)

### <font color="deepskyblue">手动安装 kubectl 套件</font>
kubectl 是 kubernetes controller， 在未来的学习笔记中，我们也常常透过 kubectl 指令 存取 kubernetes 上的物件

1. 下载套件
   ```
   $ curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/darwin/amd64/kubectl
   ```

2. 赋予执行权限
   ```
   $ chmod +x ./kubectl
   ```

3. 将 kubectl 移到 path 下
   ```
   $ sudo mv ./kubectl /usr/local/kubectl
   ```

