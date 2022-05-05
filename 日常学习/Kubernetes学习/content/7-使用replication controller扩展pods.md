本章介绍 利用 replication controller 来扩展和管理pod
+ 介绍 stateful app 和 stateless app
+ 什么是 replication controller
+ 通过 kubectl 操作 replication controller
 
## <font color=coral>什么是 stateless 和 stateful</font>
stateless 是指 应用不受时间，设备等外部条件限制， 不会影响响应返回的数据
```js
function int sum(int a, int b) {
    return a + b;
}
```
stateful 是指会记录每一步操作的状态，即使重启服务，仍然会保留， 比如 mysql
```js
int count = 0;
function int counter() {
	count++;
	return count;
}
```

## <font color=coral>replication controller 是什么</font>
replication controller 是 kubernetes 上 用来管理pod 数量和状态的 controller， 使用 replicarion controller 在 kubernetes 主要做以下几件事：
+ 每个 replication controller 都有属于自己的 yaml 文件
+ 在 yaml 中可以指定同时有多少个相同的pods运行在 Kubernetes Cluster 上
+ 当 某个 pod 发生 crash、failed 而终止运行时， replication controller 会帮助我们自动检测失败的pod， 创建新的pod， 确保pod 运行的数量和配置文件中指定的相同
+ 当设备重启时， 之前运行的 replication controller 会被自动创立， 确保 pod 随时都在运行

下面以 `my-replication-controller.yaml` 为例
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-replication-controller
spec:
  replicas: 3
  selector:
    app: hello-pod-v1
  template:
    metadata:
      labels:
        app: hello-pod-v1
    spec:
      containers:
      - name: my-pod
        image: zxcvbnius/docker-demo
        ports:
        - containerPort: 3000
```
`replication controller` 的 spec 与 pod 的 yaml 有些不同
+ spec.replicas & spec.selector  
  在`spec.replicas` 中， 我们必须定义pod的数量， 以及在 `spec.selector` 中指定我们要选择的pod的条件
+ spec.template  
  在 `spec.template` 中定义 pod 的信息，包括 pod 的 label 和 pod 中要运行的 container
+ spec.template.metadata  
  metadata 中包含 pod 的 label， metadata.labels 必须被包含在 selector 中，否则在创建 RC 时就会发生错误
+ spec.template.spec  
  最后spec的部分定义了 container, 可以参考 pod 的 yaml 文档， 在我们的例子中， 一个 pod 只有一个 container

## <font color=coral>用kubectl创建replication controller 实例</font>
```
### 创建 新的 replication controller
kubectl create -f ./sss.yaml

### 查看 创建的 replication controller
kubectl get rc

### 查看 pod 的 内容
kubectl describe pod my-replication-controller-4ftnj

### 扩展或者缩减 pod 的数量
kubectl scale --replicas=4 -f ./my-replication-controller.yaml

### 查看 rc
kubectl get rc

### 查看某个 rc 的具体信息
kubectl describe rc my-replication-controller

### 删除 rc， 对应的 pod 也会停止运行
kubectl describe rc my-replication-controller

### 删除 rc， 对应的 pod 不停止运行
kubectl delete rc my-replication-controller --cascade=false
```