以下面这个目录为例
```
└── src
    ├── CMakeLists.txt (即为 c1)
    ├── common
    │   ├── CMakeLists.txt (即为 c2)
    │   ├── package.xml
    │   └── routing
    │       ├── CMakeLists.txt
    │       ├── mm1.cpp
    │       └── mm2.cpp
    └── map
        ├── CMakeLists.txt
        ├── client
        │   ├── a_test.cpp
        │   └── b_test.cpp
        ├── cmake
        │   ├── FindGRPC.cmake
        │   └── FindProtobuf.cmake
        ├── include
        │   ├── c.h
        │   ├── d.h
        │   └── e.h
        └── src
            ├── x.c
            ├── y.c
            └── z.c
```

## c1
```python
# 指定cmake的最小版本
cmake_minimum_required(VERSION 3.2)

# 指定项目的名成， 方便其他模块的引用
project(server)

set(CMAKE_CXX_FLAGS "--std=c++14 -fext-numeric-literals -pipe -O2 -Wextra -fopenmp -fPIC -pthread")

set(PROJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)

find_package(catkin REQUIRED COMPONENTS
            common
            roscpp
            map
)

catkin_package(INCLUDE_DIRS ${PROJECT_INCLUDE_DIRS}
              LIBRARIES
              server
)

link_directories(INCLUDE_DIRS ${PROJECT_INCLUDE_DIRS}
                LIBRARIES
                server
)

# 该指令的作用主要是指定要链接的库文件的路径，该指令有时候不一定需要。因为find_package和find_library指令可以得到库文件的绝对路径。不过你自己写的动态库文件放在自己新建的目录下时，可以用该指令指定该目录的路径以便工程能够找到。
link_directories(${CATKIN_DEVEL_PREFIX}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# 使用 find_package 引入外部依赖包
# 为了方便我们在项目中引入外部依赖包，cmake官方为我们预定义了许多寻找依赖包的Module，他们存储在path_to_your_cmake/share/cmake-<version>/Modules目录下。每个以Find<LibaryName>.cmake命名的文件都可以帮我们找到一个包。
find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)
find_package(Threads)

# 添加 protobuf 的头文件目录
include_directories(${PROTOBUF_INCLUDE_DIRS})

# 添加头文件目录 
include_directories(${PROJECT_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CATKIN_DEVEL_PREFIX}/include)

FILE(GLOB_RECURSE PROTOS_ALL RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/.. *.proto)


catkin_workspace()
```

## c2
```python
cmake_minimum_required(VERSION 2.8.3)

project(map)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-mavx2 -mno-avx512f -pipe -O2 -Wall -Wextra -fopenmp -fPIC -pthread")

set(PROJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)


add_definitions(-DTEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/conf/")

find_package(catkin REQUIRED COMPONENTS
             common
             roscpp
)

catkin_package(INCLUDE_DIRS ${PROJECT_INCLUDE_DIRS}
               LIBRARIES
               map_common
               map_hdmap
               map_client_v1
               router_v2
               map_core
               map_refline
               map_refline_v2
               map_processor
               map_view
)              

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

include_directories(/opt/allride/third_party)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS iostreams)
include_directories(${Boost_INCLUDE_DIR})

include_directories(${PROJECT_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})

install(DIRECTORY ${PROJECT_INCLUDE_DIRS}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

add_subdirectory(tool)
add_subdirectory(visualizer)
```