# 通过 Ingress Controller 实现 gRPC 服务访问
[参考链接](https://help.aliyun.com/document_detail/313328.html?spm=5176.smartservice_service_robot_chat_new.help.dexternal.7a10f625A9jBFN)

远程过程调用 grpc 是基于HTTP/2 协议标准和 protobuf 序列化协议开发设计，且支持众多开发语言， 继而提供了连接多路复用、头部压缩、流控等特性， 极大的提高了客户端与服务器的通信效率。

## gRPC 服务示例
定义如下 grpc 服务， 客户端可以调用 helloworld.Greeter 服务的 SayHello 接口
```yaml
option java_multiple_files = true;
option java_package = "io.grpc.example.helloworld";
option java_outer_classname = "HelloWorldProto";

package helloworld;

service Greeter {
    rpc SayHello (HelloRequest) return (HelloReply) {}
}

message HelloRequest {
    string name = 1;
}

message HelloReply {
    string message = 1;
}
```

## 示例说明
NGINX Ingress Controller 中， gRPC 服务只运行在 HTTPS 端口 (默认端口 443) 上， 因此在生产环境中， 需要域名和应对的SSL证书

### 步骤一：申请SSL 证书
使用 Ingress 转发 gRPC 服务需要对应域名拥有 SSL 证书，使用 TLS 协议进行通信

示例使用 OpenSSL 生成的自签证书

1. 复制以下内容并保存至 /tmp/openssl.cnf 文件中
```shell
[ req ]
# default_bits
```

### 步骤二：创建 gRPC 服务所需资源
在集群中创建 gRPC 协议的后端服务， 本示例使用 镜像 `registry.cn-aliyuncs.com/acs-sample/grpc-server` 创建 grpc 服务
1. 复制以下 YAML 内容创建 grpc.yaml 文件
```yml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: grpc-service
spec:
    replicas: 1
    selector:
        matchLabels:
            run: grpc-service
    template: 
        metadata: 
            labels:
                run: grpc-service
        spec:
            containers:
            - image: registry.cn-aliyuncs.com/acs-sample/grpc-server
              imagePullPolicy: Always
              name: grpc-service
              ports:
              - containerPort: 50051
                protocol: TCP
            restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata: 
    name: grpc-service
spec:
    ports:
    - port: 50051
      protocol: TCP
      targetPort: 50051
    selector:
        run: grpc-service
    sessionAffinity: None
    type: NodePort
```

2. 执行以下命令创建 grpc deployment 和 service
```
kubectl apply -f grpc.yaml
```

### 步骤三： 创建 Ingress 路由规则
1. 复制以下 YAML 内容创建 grpc-ingress.yaml
<font color="pink">
-----------------------
+ 注意
    + 部署 gRPC 服务所使用的 ingress 需要在 annotation 中 加入 `nginx.ingress.kubernetes.io/backend-protocol`, 值为 GRPC
    + 本示例使用的域名为 grpc.example.com, 请根据实际情况修改
-----------------------
</font>

```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
    name: grpc-ingress
    annotations:
        nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
spec:
    # 指定证书
    tls:
    - hosts:
        - grpc.example.com
        secretName: grpc-secret
    rules:
    # grpc 域名服务
    - host: grpc.example.com
      http:
        paths:
        - path: /
          pathType: Prefix
          backend:
            service:
                # grpc服务
                name: grpc-service
                port:
                    number: 50051
```

2. 执行以下命令创建 Ingress 路由规则
```
kubectl apply -f grpc-ingress.yaml
```

## 结果验证
本地安装 gRPCurl 工具后， 输入 grpcurl <域名>:443 list验证请求是否成功转发到后端服务
本示例中使用域名 grpc.example.com 以及自签证书， 执行以下命令验证请求是否成功转发到后端服务
```
grpcurl -insecure -authority grpc.example.com <ip_address>:443 list
```


<font color="pink">

-----------------------
**说明**：ip_address 为 Nginx Ingress Controller 的 Service 外部 IP， 可通过 kubectl get ingress 获取

-----------------------
</font>

预期输出
```
grpc.reflection.v1alpha.ServerReflection
helloworld.Greeter
```
