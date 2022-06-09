虽然 replication controller 可以帮助我们解决很多问题， 但实际上，应用常常遇到 rollout 和 rollback 的清醒， 这就需要用到 deployment 来解决实际上的一些问题

若是回看 devops 的历史， 可以发现 devops 与 agile敏捷开发 可以说是密不可分

Kubernetes 提供了 deployment 原件， 不只帮我们做到 pod scaling， 对于 服务的 rollout 和 rollback 有很好的支持

## <font color=coral>Replica Set</font>
Replica Set 可以说是 Replication Controller 的 进化版， 与 Replication Controller 的最大不同在于 Replica Set 提供了 更弹性的 selector

以 my-replica-set.yaml 为例
```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
    name: my-replica-set
spec:
    replicas: 3
    selector:
        matchLabels:
            env: dev
        matchExpressions:
            - {key: env, operator: In, values: [dev]}
            - {key: env, operator: NotIn, values: [prod]}
    template:
        metadata:
            labels:
                app: hello-pod-v1
                env: env
                version: v1
        spec:
            containers:
                -name: my-pod
                    image: zxcbnius/docker-demo
                    ports:
                        - containerPort: 3000
```

不同于 RC 的 selector 只能用等于号表示， Replica Set 的Deployment yaml 檔的寫法與 Replica Sets 相似，如果kubectl的版本 >= 1.9，則需使用app/v1；如果版本號是在1.9以前的話，則需使用apps/v1beta2，可以用kubectl version查看目前的版本號， selector 支援更多复杂的条件过滤
+ apiVersion  
  如果 kubectl 的版本 >= 1.9, 则需使用 app/v1; 如果版本好是在 1.9 以前的话， 则需使用 apps/v1beta2
+ spec.selector.matchLabels  
  在 Replica Set 的 selector 里面提供了 matchLabels, matchLabels 的用法代表这等于(equivalent), 代表Pod的labels必须与 matchLabels 中 指定的值相同， 才算符合条件
+ spec.selector.matchExpressions  
  而 matchExpressions 的用法比较弹性， 每一条条件主要由三个部分组成key，operator， value。以 my-replica-sets.yaml 中为例， 指定 pod 的条件为 1) env 不能为 dev 2) env 不能为 prod。而目前 operator 支持四种条件 In, NotIn, Exists, 以及 DoesNotExist， 更多关于 matchExpressions 的运用可以参考官方文件 [官方文件](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)

Replica Set 与昨天提到的Replication Controller 的kubectl指令相似，可以参考第7章

而在 Kubernetes官方文件 中也提到， 虽然 Replica Set 提供更弹性的 selector， 并不推荐开发者直接使用 kubectl create 等指令创建 Replica Set 物件， 而是 透过Deployment 来创建新的 Replica Set

## <font color=coral>介绍Deployment</font>
Deploument 可以帮助我们达成以下几件事情
+ 部署一个 服务
+ 协助 applications 升级到某个特定版本
+ 服务升级过程中做到 无停机服务转移
+ 可以 rollback 到先前版本

以下是今天会用到的 my-deployment.yaml
```yaml
apiVersion: apps/v1beta2 # for kubectl versions >= 1.9.0 use apps/v1
kind: Deployment
metadata:
  name: hello-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-deployment
  template:
    metadata:
      labels:
        app: my-deployment
    spec:
      containers:
      - name: my-pod
        image: zxcvbnius/docker-demo:latest
        ports:
        - containerPort: 3000
```
Deployment yaml 的写法与 Replica Set 相似，如果kubectl 版本 >= 1.9，则需使用 app/v1, 如果版本小于 1.9 以前的话， 则需使用 app/v1beta2, 可以用 kubectl version 查看目前的版本号

|      Deployment相关指令      | 指令功能 |
|-----------------------------|-------------------------|
| kubectl get deployment | 取得目前Kubernetes中的deployments的信息 |
| kubectl get rs | 取得目前Kubernetes中的Replication Set的信息 |
| kubectl describe deploy <deployment-name> | 取得特定deployment的詳細信息 |
| kubectl set image deploy/ \<deployment-name> \<pod-name>: \<image-path>: \<version> | 將deployment管理的pod升級到特定image版本 |
|kubectl edit deploy \<deployment-name>|編輯特定deployment物件|
|kubectl rollout status deploy \<deployment-name>|查询目前某deployment升级狀況|
|kubectl rollout history deploy \<deployment-name>|查询目前某deployment升级的历史记录|
|kubectl rollout undo deploy \<deployment-name>|回滚Pod到先前一個版本|
|kubectl rollout undo deploy \<deployment-name> --to-revision=n|回滚Pod到某ge特定版本|