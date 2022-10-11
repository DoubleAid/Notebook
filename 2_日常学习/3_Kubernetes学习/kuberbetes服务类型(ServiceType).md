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
```
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
+ 它
## <font color=deepskyblue>3. ClusterIP (集群IP)</font>
## <font color=deepskyblue>4. ClusterIP (集群IP)</font>