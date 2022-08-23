
+ CMAKE_CURRENT_SOURCE_DIR： 当前处理的 CMakeLists.txt 所在的位置， The nearest folder that contains CMakeList.txt file with its own scope. (File do not need to contain project() command)
+ CMAKE_SOURCE_DIR: Topmost folder(source directory) that contains a CMakeList.txt file. The value never changes in different scopes.
+ PROJECT_SOURCE_DIR: The nearest folder that contains CMakeList.txt file, with its own scope, that contains project() command.
+ CMAKE_CURRENT_LIST_DIR: The folder that contains currently processed CMakeList.txt or .cmake file.


[在cpp中添加grpc](https://blog.csdn.net/fengfengdiandia/article/details/83591171)