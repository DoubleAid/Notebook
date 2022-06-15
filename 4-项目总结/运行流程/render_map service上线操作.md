### 获取 render_map 的 数据版本

数据版本的上传规则就是上传到oss://allride-map/data/render_map/{地区}/{版本号}/render_map.zip

获取 版本号 和 地区
### 查看苏州的 service 的版本号

查看 render map 的 service 服务
```
(base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Document/Notebook$ kubectl -n ridesharing get pod | grep renderrender-map-service-6788d954d6-9vrtz                         2/2     Running     2          223d
render-map-service-v2-856dc576c-qvh6f                       3/3     Running     0          6d21h
render-map-service-v2-nanjing-847599b5fb-6fs84              3/3     Running     0          3h45m
render-map-service-v2-nanjing-847599b5fb-wcw5j              3/3     Running     2          3h45m
render-map-service-v2-test-6b44cb45fc-hkpld                 3/3     Running     0          6d21h
```
其中 render-map-service-v2-test-6b44cb45fc-hkpld 为测试版本， 一般后台功能有修改，先修改测试版本，等前端测试通过后，再将主版本改过去 查看版本
```
(base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Document/Notebook$ kubectl -n ridesharing describe pod render-map-service-v2-856dc576c-qvh6f | grep grpc_service_v2
    Image:          allride-registry.cn-shanghai.cr.aliyuncs.com/map/grpc_service_v2:v0.4.2
    Image ID:       docker-pullable://allride-registry.cn-shanghai.cr.aliyuncs.com/map/grpc_service_v2@sha256:5c641b6cd0bdd941400a286f690806b783fd783dc2a7cf4c44576c4951bdb02d
```
### 修改 yaml
```
---
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: ridesharing
  name: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
  labels:
    component: ridesharing
    app: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
  template:
    metadata:
      labels:
        component: ridesharing
        app: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
    spec:
      containers:
        - name: render-map-storage
          image: allride-registry.cn-shanghai.cr.aliyuncs.com/map/render_map_storage:v0.0.1 # 修改了 后面的 版本号
          ports:
            - containerPort: 5432
              name: port
        - name: render-map-cache
          image: allride-registry.cn-shanghai.cr.aliyuncs.com/map/map_redis:v0.0.1
          ports:
            - containerPort: 6379
              name: port
        - name: render-map-grpc
          image: allride-registry.cn-shanghai.cr.aliyuncs.com/map/grpc_service_v2:v0.4.2 # 修改了这里， 修改service 的 版本号
          ports:
            - containerPort: 50051
              name: port
          env:
            - name: OSS_ACCESS_KEY_ID
              value: "LTAI5tBQ7xTg2r3tzecZQsQQ" # 添加了自己的 id
            - name: OSS_ACCESS_KEY_SECRET
              value: "qXbsp4p09yPE7r3LFqlEa21yProvqQ" # 添加了自己的 password
            - name: ALIYUN_REGION
              value: "cn-shanghai"
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  namespace: ridesharing
  name: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
spec:
  type: LoadBalancer
  ports:
    - port: 51053
      targetPort: 50051
      protocol: TCP
  selector:
    component: ridesharing
    app: render-map-service-v2-nanjing # 修改这里，添加了 nanjing
```
### 创建该服务
```
kubectl -n ridesharing create -f service-render-map-cache.yaml
```
### 查看运行结果
```
(base) guanggang.bian@G15-FNTZ3G3-U:~/Private/Document/Notebook$ kubectl -n ridesharing get svc | grep render
render-map-service                             LoadBalancer   10.129.206.223   101.132.37.205    51053:32761/TCP   320d
render-map-service-v2                          LoadBalancer   10.129.36.241    47.103.91.163     51053:32733/TCP   118d
render-map-service-v2-nanjing                  LoadBalancer   10.129.222.150   139.196.72.176    51053:31500/TCP   4h45m
render-map-service-v2-test                     LoadBalancer   10.129.217.136   47.100.24.1       51053:32150/TCP   62d
```