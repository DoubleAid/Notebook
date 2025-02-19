# Kubernetes 服务类型 (Service Types)
[<font color=coral>参考文档</font>](https://zhuanlan.zhuhu.com/p/406369532)

![](https://pic1.zhimg.com/v2-c960a6819de0aa5d5fefda8daf09fe00_r.jpg)

Kubernetes 服务有四种类型 -- ClusterIP、NodePort、LoadBalancer 和 ExternalName。服务 spec 中的 type 属性决定了服务如何暴露给网络

## <font color=deepskyblue>1. ClusterIP (集群IP)</font>
+ ClusterIP 是默认和最常见的服务类型
+ Kubernetes 会为 ClusterIP 服务分配一个集群内部IP地址。 这使得服务只能在集群内访问
+ 您不能在集群外部向服务 (pods) 发出请求
+ 您可以选择在服务定义文件中设置集群IP
### 使用场景
集群内的服务间通信。 例如，应用程序的前端（front-end）和 后端 (back-end)组件之间的通信
### 举例
```yml
apiVersion: v1
kind: Service
metadata:
    name: my-backend-service
spec:
    type: ClusterIP # Optional field (default)
    clusterIP: 10.0.0.1 # within service cluster ip range
    ports:
        - name: http
          protocol: TCP
          port: 80
          targetPort: 8080
```
## <font color=deepskyblue>2. NodePort (节点端口)</font>
+ NodePort 服务是 ClusterIP 服务的扩展。 NodePort 服务路由到的 ClusterIP 服务会自动创建
+ 它通过在 ClusterIP 之上添加一个集群范围的端口来公开集群外部的服务
+ NodePort 在静态端口 (NodePort) 上公开每个节点IP上的服务。 每个节点将该端口代理到你的服务中。 因此， 外部流量可以访问每个节点上的固定端口。这意味着该端口上的集群的任何请求都会转发到该服务
+ 你可以通过请求 <NodeIP>:<NodePort> 从集群外部联系 NodePort 服务。
+ 节点端口必须在 30000-32767 范围内， 手动为服务分配端口是可选的， 如果未定义， Kubernetes 会自动分配一个
+ 如果你要明确选择节点端口， 请确保该端口尚未被其他服务使用
### 使用场景
+ 当你想要启用与你的服务的外部联系时
+ 使用 NodePort 可以让你自由的设置自己的负载均衡解决方案， 配置 Kubernetes 不完全支持的环境， 甚至直接公开一个或多个节点的IP。
+ 最好在节点上方放置负载均衡器以避免节点故障
### 举例
```yml
apiVersion: v1
kind: Service
metadata:
    name: my-frontend-service
spec:
    type: NodePort
    selector:
        app: web
    ports:
        - name: http
          protocol: TCP
          port: 80
          targetPort: 8080
          nodePort: 30001 # 30000 - 32767 Optional field
```
## <font color=deepskyblue>3. LoadBalancer (负载均衡器)</font>
+ LoadBalancer 服务是 NodePort 服务的扩展。 外部负载均衡器路由到 NodePort 和 ClusterIP 服务是自动创建的
+ 它将 NodePort 与 基于云的负载均衡器集成在一起。
+ 它使用云厂商的负载均衡器在外部公开服务
+ 每个云厂商 (AWS, Azure, GCP, Aliyun, Tencent) 都有自己的原生负载均衡器实现。云厂商将创建一个负载均衡器， 然后它会自动将请求路由到你的 Kubernetes 服务
+ 来自外部负载均衡器的流量被定向到后端 Pod。 云厂商决定如何进行负载均衡
+ 负载均衡器的实际创建是异步发生的
+ 每次要向外部公开服务时， 都必须创建一个新的 LoadBalancer 并获取 IP 地址
### 使用场景
当你使用云厂商来托管你的 Kubernetes 集群时
### 举例
```yml
apiVersion: v1
kind: Service
metadata:
    name: my-frontend-service
spec:
    type: LoadBalancer
    clusterIP: 10.0.171.123
    loadBalancerIP: 123.123.123.123
    selector:
        app: web
    ports:
        - name: http
          protocol: TCP
          port: 80
          targetPort: 8080
```
## <font color=deepskyblue>4. ExternalName (外部名称)</font>
+ ExternalName 类型的服务将 Service 映射到 DNS 名称， 而不是典型的选择器， 例如 my-service
+ 你可以使用 `spec.externalName` 参数指定这些服务
+ 它通过返回带有其值的 CNAME 记录， 将服务映射到 externalName 字段 (例如 foo.bar.example.com) 的内容
+ 没有建立任何类型的代理

### 使用场景
+ 这通常用于在 Kubernetes 内创建服务来表示外部数据存储， 例如在 Kubernetes 外部运行的数据库
+ 当来自一个命名空间的 Pod 与另一个命名空间中的服务通信时， 可以使用该 ExternalName 服务（作为本地服务）

### 举例
```yml
apiVersion: v1
kind: Service
metadata:
    name: my-service
spec:
    type: ExternalName
    externalName: my.database.example.com
```

## <font color=deepskyblue>5. Ingress入口 </font>
你还可以使用 Ingress 来公开你的服务。 Ingress 不是服务类型， 但它充当集群的入口点。 它允许将路由规则整合到单一资源中，因为他可以在同一 IP 地址下公开多个服务

不同类型的ingress控制器有不同的功能

默认的 GKE ingress 控制器会启动一个 HTTP(S) Load Balancer, 可以通过基于路径或者是基于子域名的方式路由到后端服务。 例如， 可以通过 foo.yourdomain.com 发送任何东西到 foo 服务， 或者是发送 yourdomain.com/bar/ 路径下的任何东西到 bar 服务

对于使用 第七层 HTTP Load Balancer 的 GKE 上的 Ingress 对象， 其 YAML 文件如下
```yml
apiVersion: extensions/v1beta1
kind: Ingress
metadata: 
    name: my-ingress
spec:
    backend:
        serviceName: other
        servicePort: 8080
    rules:
      - host: foo.mydomain.com
        http:
            paths:
              - backend:
                  serviceNmae: foo
                  servicePort: 8080
      - host: foo.mydomain.com
        http:
            paths:
              - path: /bar/*
                backend:
                  serviceName: bar
                  servicePort: 8080
```

### 使用场景
Ingress 是发布功能最强大， 也是最复杂的。 Ingress 控制器的类型很多， 如 Google Cloud Load Balancer， Nginx， Contour， Istio等等。 还有一些 Ingress 控制器插件， 比如证书控制器， 可以自动为服务提供ssl认证

如果想在同一个 IP 地址下发布多个服务， 并且这些服务使用相同的第七层协议， 推荐使用 Ingress

### Ingress yaml 文件参考
```yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
    name: ingress-wildcard-host
spec:
    rules:
        - host: "foo.bar.com"
          http:
            paths:
                - pathType: Prefix
                  path: "/bar"
                  backend:
```