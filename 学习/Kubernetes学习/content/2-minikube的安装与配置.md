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

4. 通过 kubectl 命令查看可用指令
   ```
   (base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Document/Notebook$ kubectl
   kubectl controls the Kubernetes cluster manager.

   Find more information at: https://kubernetes.io/docs/reference/kubectl/overview/

    Basic Commands (Beginner):
    create        Create a resource from a file or from stdin
    expose        Take a replication controller, service, deployment or pod and expose it as a new Kubernetes service
    run           Run a particular image on the cluster
    set           Set specific features on objects

    Basic Commands (Intermediate):
    explain       Get documentation for a resource
    get           Display one or many resources
    edit          Edit a resource on the server
    delete        Delete resources by file names, stdin, resources and names, or by resources and label selector

    Deploy Commands:
    rollout       Manage the rollout of a resource
    scale         Set a new size for a deployment, replica set, or replication controller
    autoscale     Auto-scale a deployment, replica set, stateful set, or replication controller

    Cluster Management Commands:
    certificate   Modify certificate resources.
    cluster-info  Display cluster information
    top           Display resource (CPU/memory) usage
    cordon        Mark node as unschedulable
    uncordon      Mark node as schedulable
    drain         Drain node in preparation for maintenance
    taint         Update the taints on one or more nodes

    Troubleshooting and Debugging Commands:
    describe      Show details of a specific resource or group of resources
    logs          Print the logs for a container in a pod
    attach        Attach to a running container
    exec          Execute a command in a container
    port-forward  Forward one or more local ports to a pod
    proxy         Run a proxy to the Kubernetes API server
    cp            Copy files and directories to and from containers
    auth          Inspect authorization
    debug         Create debugging sessions for troubleshooting workloads and nodes

    Advanced Commands:
    diff          Diff the live version against a would-be applied version
    apply         Apply a configuration to a resource by file name or stdin
    patch         Update fields of a resource
    replace       Replace a resource by file name or stdin
    wait          Experimental: Wait for a specific condition on one or many resources
    kustomize     Build a kustomization target from a directory or URL.

    Settings Commands:
    label         Update the labels on a resource
    annotate      Update the annotations on a resource
    completion    Output shell completion code for the specified shell (bash, zsh or fish)

    Other Commands:
    alpha         Commands for features in alpha
    api-resources Print the supported API resources on the server
    api-versions  Print the supported API versions on the server, in the form of "group/version"
    config        Modify kubeconfig files
    plugin        Provides utilities for interacting with plugins
    version       Print the client and server version information

    Usage:
    kubectl [flags] [options]

    Use "kubectl <command> --help" for more information about a given command.
    Use "kubectl options" for a list of global command-line options (applies to all commands).
   ```

### <font color="deepskyblue">从Github下载minikube套件</font>
可以从 github上直接下载 [minikube](https://github.com/kubernetes/minikube)  
linux 下载参考 --> [minikube document](https://minikube.sigs.k8s.io/docs/start/)
对于 macOS 可以通过brew安装套件
```
$ brew cask install minikube
```
安装后通过 minikube 查看
```
(base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Document/Notebook$ minikube
minikube provisions and manages local Kubernetes clusters optimized for development workflows.

Basic Commands:
  start          Starts a local Kubernetes cluster
  status         Gets the status of a local Kubernetes cluster
  stop           Stops a running local Kubernetes cluster
  delete         Deletes a local Kubernetes cluster
  dashboard      Access the Kubernetes dashboard running within the minikube cluster
  pause          pause Kubernetes
  unpause        unpause Kubernetes

Images Commands:
  docker-env     Configure environment to use minikube's Docker daemon
  podman-env     Configure environment to use minikube's Podman service
  cache          Add, delete, or push a local image into minikube
  image          Manage images

Configuration and Management Commands:
  addons         Enable or disable a minikube addon
  config         Modify persistent configuration values
  profile        Get or list the current profiles (clusters)
  update-context Update kubeconfig in case of an IP or port change

Networking and Connectivity Commands:
  service        Returns a URL to connect to a service
  tunnel         Connect to LoadBalancer services

Advanced Commands:
  mount          Mounts the specified directory into minikube
  ssh            Log into the minikube environment (for debugging)
  kubectl        Run a kubectl binary matching the cluster version
  node           Add, remove, or list additional nodes
  cp             Copy the specified file into minikube

Troubleshooting Commands:
  ssh-key        Retrieve the ssh identity key path of the specified node
  ssh-host       Retrieve the ssh host key of the specified node
  ip             Retrieves the IP address of the specified node
  logs           Returns logs to debug a local Kubernetes cluster
  update-check   Print current and latest version number
  version        Print the version of minikube
  options        Show a list of global command-line options (applies to all commands).

Other Commands:
  completion     Generate command completion for a shell

Use "minikube <command> --help" for more information about a given command.
```

启动 minikube 之后， 会在 用户 home 目录下 多出一个 `~/.kube` 的文件夹， 而 `kubectl` 就是通过该文件夹下的 configuration 与 minikube 沟通， 可以用 cat 查看 `~/.kube/config` 的内容

最后， 可以通过 `minikube status` 查看当前的状态

### <font color="deepskyblue">在 minikube 上执行 hello-minikube app </font>
启动 minikube 之后，可以通过 `kubectl run` 在 minikube 上运行一个 Google 提供的 hello-minikube docker image, 输入以下指令
```
$ kubectl run hello-minikube --image=gcr.io/google_containers/echoserver:1.8 --port=8080
deployment "hello-minikube" created
```
然后执行 `kubectl expose` 指令， 让本地链接到 `hello-minikube` 服务
```
(base) guanggang.bian@G15-FNTZ3G3-U:~$ kubectl get pod
NAME              READY   STATUS    RESTARTS   AGE
hello-minikube    1/1     Running   0          14m
hello-minikubes   1/1     Running   0          5m43s
(base) guanggang.bian@G15-FNTZ3G3-U:~$ kubectl expose pod hello-minikube --type=NodePort
service/hello-minikube exposed
```
通过 `minikube service hello-minikube --url` 去获得这个 service 的 url
```
(base) guanggang.bian@G15-FNTZ3G3-U:~$ minikube service hello-minikube --url
http://192.168.59.100:31000
```
可以通过 url 查看 service 的运行状态
+ 每次产生的url是由系统决定的
+ 可以尝试在url后面带入不同的参数，可以看到real path 的转换， 例如 http://192.168.x.x:xxxx/hellokube
