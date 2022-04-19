### 摘要
+ 认识 kubernetes 最小的运行单位 pod
+ 如何与 pod 中的 container 互动
+ 常见的 kubectl 指令

## <font color="coral">认识 Pod</font>
在使用 kubernetes 运行 docker container 之前，需要先认识一下 pod

### <font color="deepskyblue">pod是很么</font>
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

这样就建立好了 my-app 这个 pod 了， 可以用 `kubectl get pods` 查看
