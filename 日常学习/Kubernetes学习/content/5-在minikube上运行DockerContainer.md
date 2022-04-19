### 摘要
+ 认识 kubernetes 最小的运行单位 pod
+ 如何与 pod 中的 container 互动
+ 常见的 kubectl 指令

## <font color="coral">认识 Pod</font>
在使用 kubernetes 运行 docker container 之前，需要先认识一下 pod

### <font color="deepskyblue">pod是什么</font>
Kubernetes就相当于一个手机，上面会运行很多不同类型的app， 而一个Pod在Kubernetes就相当于一个app
pod有这些特点：
+ 每个 pod 都有属于自己的 yaml
+ 一个 pod 里面可以包含一个或者多个docker container
+ 在同一个 pod 里面的 container， 可以用 local port numbers 来相互沟通

### <font color="deepskyblue">如何建立一个 pod</font>
```yaml
# my-first-pod.yaml
apiVersion: v1
kind: Pod
metadata:
    name: my-pod
    labels:
    app: webserver
spec:
    containers:
    - name: pod-demo
      image: doubleaid/docker-demo
      ports:
      - containerPort: 3000
```
+ apiVersion
  apiVersion 表示当前 Kubernetes 的 pod 版本， 你可以查看 `~/.kube/config` 内的版本
  v1 是 目前 Kubernetes 中的核心版本
+ metadata
  在metadata中， 有三个重要的key， 分别是 name， label， annotations
  + metadata.name
    在 `metadata.name` 位置指定 pod 的名称
  + metadata.labels
    `metadata.labels` 是 Kubernetes 的核心的元件之一， Kubernetes会通过 label selector 将 pod 分群管理
  + metadata。annotations
    annotation 的功能和labels相似， annotation 通常是由开发者自定义的附加信息， 例如 版本信息，发布日期等等

+ spec
  最后的 spec 部分则是定义container
  + container.name： 设置 container 的名称
  + container.image : Image 则是根据 docker registry 提供的可下载路径
  + container.ports ： 最后 ports 的部分则是可以指定该 container 的那些 port 是允许访问的

在了解和 yaml 的每一行都在干什么事情之后， 可以使用 `kubectl create` 指令在 `kubernetes cluster` 中建立 pod 物件
```shell
$ kubectl create -f my-first-pod.yaml
pod "my-pod" created
```

这样就建立好了 my-app 这个 pod 了， 可以用 `kubectl get pods` 查看目前 pods 的状态
```
$ kubectl get pods
NAME    READY    STATUS              RESTARTS   AGE
my-pod  0/1      ContainerCreating   0          41s
```
可以看到 `ContainerCreating` 的状态， 如果再等一下状态酒水变成 `Running`， 表示 pod 已经正常运行了

可以用 `kubectl describe` 查看 my-pod 的详细信息， 包括 pod name， labels， 以及这个pod的历史状态， 更多的 log 在 describe-my-pod.log
```
$ kubectl describe pods my-pod
Name:         my-pod
Namespace:    default
.....
Events:
  Type    Reason                 Age   From               Message
  ----    ------                 ----  ----               -------
  Normal  Scheduled              7m    default-scheduler  Successfully assigned my-pod to minikube
  Normal  SuccessfulMountVolume  7m    kubelet, minikube  MountVolume.SetUp succeeded for volume "default-token-wxjzb"
  Normal  Pulling                7m    kubelet, minikube  pulling image "zxcvbnius/docker-demo"
  Normal  Pulled                 7m    kubelet, minikube  Successfully pulled image "zxcvbnius/docker-demo"
  Normal  Created                7m    kubelet, minikube  Created container
  Normal  Started                7m    kubelet, minikube  Started container
```

### <font color="deepskyblue">与 pod 中的 container 进行交互</font>
在 my-pod 中， 有 container 在运行 api server， port num 3000， 那么如何和 container 进行交互呢，
#### <font color="GreenYellow">方法一 通过kubectl port-forward</font>
kubectl 提供 `port-forward` 指令 能将 pod 中的耨个 port number 与本机的 port 做 mapping， 
```
$ kubectl port-forward my-pod 8000
Forwarding from 127.0.0.1:8000 -> 3000
```
看到 `Forwarding from 127.0.0.1:8000 -> 3000` 代表 port-forward 成功， 现在可以在本地的浏览器`127.0.0.1:8000`访问

#### <font color=greenyellow>方法二 建议一个 service</font>
通过 `kubectl expose` 在 kubernetes 建立一个service服务

kubectl port-forward 是将pod 的 port 映射到本地， 而 `kubectl expose` 则是将 pod 的 port 和 kubernetes cluster 的 port 做映射

首先使用 minikube status 查看目前 minikube-vm  使用的哪个内部网址
```
$ minikube status
minikube: Running
cluster: Running
kubectl: Correctly Configured: pointing to minikube-vm at 192.168.99.104
```

