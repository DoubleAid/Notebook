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
+ spec.template
+ spec.template.metadata
  
+ spec.template.spec
  最后spec的部分定义了 container, 可以参考 pod 的 yaml 文档， 在我们的例子中， 一个 pod 只有一个 container