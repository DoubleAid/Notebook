+ 首先在 src/map/processor 下新建一个测试文件 test.cpp
+ 在 src/map/processor/CMakeLists.txt 中添加 text.cpp 的相关配置
  ```cpp
  // 添加 可执行文件
  add_executable(bgg_map_test_main bgg_map_test_main.cpp)

  // 添加链接库
  target_link_libraries(bgg_map_test_main
                      map_common
                      map_hdmap
                      map_refline
                      map_refline_v2
                      map_processor
                      router_v2
                      pqxx
                      pq
                      ${PROTOBUF_LIBRARIES}
                      ${catkin_LIBRARIES}
    )
  
  // 在 install targets 添加可执行文件
  install(TARGETS map_builder bgg_map_test_main
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  ```

+ 编译，编译过程可以参考工程根目录下的readme
+ 运行数据库 docker
    ```
    sudo docker run allride-registry.cn-shanghai.cr.aliyuncs.com/map/map_storage:v2_8.0.100

    // 通过 docker inspect 查看 ip 地址
    sudo docker ps
    sudo docker inspect --format='{{.NetworkSettings.IPAddress}}' b78df4b125ec
    ```

+ 下载相应的cfg文件, 并根据上面获取的ip地址进行修改
  cfg文件格式如下：
    ```
    client {
    local_map_file: "/opt/allride/data/map/hdmap_no_roi.pb.txt"
    }

    road_map {
    client {
    db {
      host: "172.17.0.2"
      port: 5432
      db: "map"
      table: "road_map"
    }
    cache {
      radius: 9
      size: 375
    }
    }
    }

    environment_map {
        client {
            db {
                host: "172.17.0.2"
      port: 5432
      db: "map"
      table: "environment_map"
    }
    cache {
      radius: 4
      size: 64
    }
    }
    }

    router {
    client {
    db {
      host: "172.17.0.2"
      port: 5432
      db: "map"
      table: "road_map"
    }
    cache {
      radius: 9
      size: 375
    }
    }
    cost {
    change_lane_cost: 20
    left_turn_cost: 50
    right_turn_cost: 10
    u_turn_cost: 50
    }
    }

    road_map_smoothed {
    client {
    db {
      host: "172.17.0.2"
      port: 5432
      db: "map"
      table: "road_map_smoothed"
    }
    cache {
      radius: 9
      size: 375
    }
    }
    }
    ```
+ 编译出来的文件在 devel/lib/map 中，
