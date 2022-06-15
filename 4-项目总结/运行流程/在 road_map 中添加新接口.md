[地图 grpc 服务文档](https://allride.yuque.com/xgdyp5/gzgih4/hdo473)

## <font color=coral>步骤列表</font>
修改 `src/server/proto/road_map_service.proto` 在下方添加四个message
```protobuf
message GetTileClientData {
  GeoTileId id = 1;
}

message GetTileServiceData {
  HDMapTile tile = 1;
}

message GetTileRequest {
  RequestHeader header = 1;
  GetTileClientData data = 2;
}

message GetTileResponse {
  ResponseHeader header = 1;
  GetTileServiceData data = 2;
}
```
在 `service RoadMapService` 中 添加接口
```
rpc GetTile(GetTileRequest) returns (GetTileResponse);
```

## <font color=coral>修改 road_map_call.h </font>
在`road_map_call.h`中添加`GetTileCall`
```cpp
class GetTileCall : public RoadMapCall<proto::map::GetTileRequest, proto::map::GetTileResponse> {
public:
  void Proceed();
  // 构造函数
  GetTileCall(std::shared_ptr<hdmap_v2::RoadMap>& road_map,
                 proto::map::RoadMapService::AsyncService* service, 
                 ::grpc::ServerCompletionQueue* cq)
     : RoadMapCall(service, cq) {
    road_map_ = road_map;
    Proceed();
  }
private:
  std::vector<hdmap_v2::SignalRuleConstPtr> signals_;
};
```
## <font color=coral>实现 proceed 方法 </font>
在 `road_map_call.cc` 中实现 相应的 processed 方法
```cpp
void GetTileCall::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    service_->RequestGetTile(&ctx_, &req_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    VINFO("RoadMap Service GetTileCall rpc");
    VINFO("Request id: " << req_.header().id() << 
          "  tile id: " << req_.data().id().x() << ", "
          << req_.data().id().y() << ", "
          << req_.data().id().z();
    new GetTileCall(road_map_, service_, cq_);
    const auto& tile_data = road_map_->getTile(GeoTileId(req_.data().id().x(),req_.data().id().y(),req_.data().id().z()));
    rep_.mutable_header()->set_id(req_.header().id());
    if (tile_data != nullptr) {
      rep_.mutable_header()->set_status_code(proto::map::StatusCode::SUCCESS);
      rep_.mutable_header()->set_error_msg("none");
    } else {
      rep_.mutable_header()->set_status_code(proto::map::StatusCode::SUCCESS);
      rep_.mutable_header()->set_error_msg("error");
    }
    status_ = FINISH;
    responder_.Finish(rep_, ::grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}
```

## <font color=coral>修改 config_grpc </font>
在 `RoadMapRpcs` 中添加一个标记位
```

```
## 添加 map_storage
```
docker run -idt -p 5432:5432 allride-registry.cn-shanghai.cr.aliyuncs.com/map/map_storage:v2_0.1.61
```
## <font color=coral>编译镜像 </font>
```
sudo docker build --rm -f Dockerfile_service --build-arg JFROG_USERNAME=guanggang.bian --build-arg JFROG_APIKEY=AP55ACrFS3jQsUxsPxY1AU5Ka3J -t allride-registry.cn-shanghai.cr.aliyuncs.com/map/grpc_service_v2:vget_tile .
```
## <font color=coral>上传镜像</font>
```
sudo docker push allride-registry.cn-shanghai.cr.aliyuncs.com/map/grpc_service_v2:vget_tile 
```

## <font color=coral>修改 yaml 文件</font>

## <font color=coral>创建 deployment</font>
kubectl -n ridesharing create -f xxxxxx.yaml

## <font color=coral> 查看pod的错误信息 <font>
查看container 的错误信息
```
kubectl -n ridesharing logs -f $pod_name -c $container_name
```

## <font color=coral> 修改后应用该 yaml 文件 </font>
```
kubectl -n ridesharing apply -f grpc_get_tile.yaml
```

## <font color=coral> 获取接口信息 </font>
```
kubectl -n ridesharing get svc | grep get-tile
```
## <font color=coral> 删除 deployment </font>
 kubectl -n ridesharing delete deployment map-xxxxxxxx

## 修改 src/common/proto/map/map_service.proto 