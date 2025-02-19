### 摘要
+ 介绍 Node 是什么
+ 简述 Kubernetes 的内部运作

## <font color="coral">Node 是什么</font>
在 Kubernetes 中， Node 通常是指实体机或者虚拟机，一个 node 可以指 AWS 的一台 EC2 或者 GCP上的一台 computer engine，或者是实际中的计算机， 只要上面装有 docker engine， 足以跑起 pod， 就可以加入 kubernetes cluster

将 node 加到 kubernetes 中之后， kubernetes 会建立一个 node 节点， 并进行一系列检查， 包括网络连接， pod 能否正常启动， 若都通过 node 的状态 就会设为 ready

通过 `kubectl get nodes` 查看 node 的信息
```
(base) $ kubectl get nodes
NAME                        STATUS   ROLES    AGE    VERSION
195   Ready    <none>   661d   v1
198   Ready    <none>   660d   v1
199   Ready    <none>   660d   v1
```
## <font color="coral">用 Kubernetes 管理 Nodes 的好處</font>
过去 一台 node 节点 只会运行一个 app， 如果 应用 使用了 io，就会使内存闲置。为了避免资源浪费， 会将多个服务同时运行在 一个 node 上, 但需要时刻见识应用的资源使用情况

在 Kubernetes 上， 当我们把 node 加入到 kubernetes cluster 后， 系统会根据当前 pod 的 设定去决定部署在那个 node 上面
## <font color=coral>Kubernetes 的 内部运行</font>
[参考资料](https://ithelp.ithome.com.tw/articles/10193248)
