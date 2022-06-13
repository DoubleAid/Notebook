我们需要在pod前面再接一层桥梁， 确保每次存取服务时，都能连接到正在运行的pod
在前几章使用 `kubectl expose`, 该指令可以帮我们撞见一个新的 Service 物件， 来让 Kubernetes Cluster 中运行的 Pod 与外部相互沟通

一个 Service 可以为 Pods 做到以下几个事情

+  创建一个 ClusterIP， 让 Kubernetes Cluster 中的其他服务