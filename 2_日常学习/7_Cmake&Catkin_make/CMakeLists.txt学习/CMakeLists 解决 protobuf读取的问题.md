```python
project(grpc_service)

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# 设置输出路径
SET (PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
if (NOT EXISTS "${PROTO_OUTPUT_DIR}" OR NOT IS_DIRECTORY "${PROTO_OUTPUT_DIR}")
    file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})
endif()

# 设置 protoc 的搜索路径
LIST(APPEND PROTO_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/proto)

# 获取需要编译的 proto 文件
file(GLOB_RECURSE PROTOS_ALL ${CMAKE_CURRENT_SOURCE_DIR}/proto/*.proto)

set(PROTO_SRCS "")
set(PROTO_HDRS "")

foreach(proto_file ${PROTOS_ALL})
    get_filename_component(FIL_WE ${proto_file} NAME_WE)

    list(APPEND PROTO_SRCS "${PROTO_OUTPUT_DIR}/${FIL_WE}.pb.cc")
    list(APPEND PROTO_HDRS "${PROTO_OUTPUT_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/proto/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/proto/${FIL_WE}.pb.h"
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --cpp_out ${PROTO_OUTPUT_DIR}
             -I ${CMAKE_CURRENT_SOURCE_DIR}/proto
             ${proto_file}
        DEPENDS ${proto_file}
        COMMENT "Running C++ protocol buffer compiler on ${proto_file}"
        VERBATIM
    )
endforeach()

# 设置文件属性为 GENERATED
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

add_custom_target(generate_message ALL
                  DEPENDS ${PROTO_SRCS} ${PROTO_HDRS}
                  COMMENT "generate message target"
                  VERBATIM
)               

include_directories(./include ${PROTO_OUTPUT_DIR})

add_executable(service service.cpp
    ${PROTO_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server_implement.cpp
)

target_link_libraries(service
                      ${Protobuf_LIBRARIES}
)
```