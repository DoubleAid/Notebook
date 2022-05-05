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
```