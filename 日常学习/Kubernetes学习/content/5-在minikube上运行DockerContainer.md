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
接着输入 `kubectl expose` 指令， 创建一个名为 `my-pod-service` 的Service服务
```
$ kubectl expose pod my-pod --type=NodePort --name=my-pod-service
service "my-pod-service" exposed
```

这时可以输入 `kubectl get services` 查看目前运行的 service
```
$ kubectl get services
NAME             TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
kubernetes       ClusterIP   10.96.0.1      <none>        443/TCP          9h
my-pod-service   NodePort    10.97.118.40   <none>        3000:30427/TCP   1m
```
可以看到 `my-pod-service` 将 `my-pod` 的 port number 3000 与 minikube-vm 上的 port number 30427 做映射。接着用 `minikube service` 快速找到 `my-pod-service` 的 url
```
$ minikube service my-pod-service --url
http://192.168.99.104:30427
```
可以通过这个 url 访问 api 了

### <font color=deepskyblue>常见的kubectl指令</font>
#### <font color=greenyellow>取得 kubernetes cluster 中所有正在运行的pods信息</font>
```
$ kubectl get pods
# 如果加上 --show-all， 会显示所有的 pods
kubectl get pods --show-all
```
#### <font color=greenyellow>获取某个pod的详细信息</font>
```
kubectl describe pod <pod>
```
#### <font color=greenyellow>将某个 Pod 中指定的 port number expose 出來讓外部可以存取(建立一個新的 Service)</font>
```
$ kubectl expose pod <pod> --port=<port> --name=<service-name>
```
#### <font color=greenyellow>将某个 Pod 中指定的 port number 映射到本機端的某一特定 port number</font>
```
$ kubectl port-forward <pod> <external-port>:<pod-port>
```
#### <font color=greenyellow>当一个container起来之后， 有时希望能进到 container 内部去看 logs， 可以使用 kubectl attach 指令</font>
```
kubectl attach <pod> -i
```
#### <font color=greenyellow>可以对pod执行内部指令</font>
```
kubectl exec <pod> -- <command>
```
比如查看 my-pod 的 /app 文件夹下的 api server 的源文件，
```
kubectl exec my-pod -- ls /app
Dockerfile
index.js
node_modules
package.json
```
#### <font color=greenyellow>可以新增 Pod 的 Labels</font>
```
$ kubectl label pods <pod> <label-key>=<label-value>
```
可以用 `kubectl get pod --show-labels` 查看目前 my-pod 有那些 labels
```
$ kubectl get pods  --show-labels
NAME      READY     STATUS    RESTARTS   AGE       LABELS
my-pod    1/1       Running   0          14h       app=webserver
```
如果想要新增 labels， 可以输入以下指令
```
$ kubectl label pods my-pod version=latest
pod "my-pod" labeled
```
再用 `kubectl get pod --show-labels` 查看
```
$ kubectl get pod --show-labels
NAME      READY     STATUS    RESTARTS   AGE       LABELS
my-pod    1/1       Running   0          14h       app=webserver,version=latest
```
#### <font color=greenyellow>使用 alpine 查看 cluster 狀況</font>
alpine 提供非常轻量级的 dockerimage， 大小只用5MB。 可以借由在 alpine 下指令， 在 Kubernetes Cluster 中与其他 pod 互动，适合用来 debug，
```
$ kubectl run -i --tty alpine --image=alpine --restart=Never -- sh
```
#### <font color=greenyellow></font>