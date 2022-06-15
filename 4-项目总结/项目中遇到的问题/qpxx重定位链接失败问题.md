### 问题描述
编译错误信息如下
```
[ 79%] Linking CXX shared library /home/guanggang.bian/Private/Code/common/devel/lib/libmap_client_v1.so
/usr/bin/ld: //usr/local/lib/libpqxx.a(connection.o): relocation R_X86_64_PC32 against symbol `_ZTVN4pqxx16connectionpolicyE' can not be used when making a shared object; recompile with -fPIC
/usr/bin/ld: final link failed: Bad value
collect2: error: ld returned 1 exit status
map/client/hd_map/CMakeFiles/map_view.dir/build.make:516: recipe for target '/home/guanggang.bian/Private/Code/common/devel/lib/libmap_view.so' failed
make[2]: *** [/home/guanggang.bian/Private/Code/common/devel/lib/libmap_view.so] Error 1
CMakeFiles/Makefile2:4561: recipe for target 'map/client/hd_map/CMakeFiles/map_view.dir/all' failed
make[1]: *** [map/client/hd_map/CMakeFiles/map_view.dir/all] Error 2
make[1]: *** Waiting for unfinished jobs....
/usr/bin/ld: //usr/local/lib/libpqxx.a(connection.o): relocation R_X86_64_PC32 against symbol `_ZTVN4pqxx16connectionpolicyE' can not be used when making a shared object; recompile with -fPIC
/usr/bin/ld: final link failed: Bad value
collect2: error: ld returned 1 exit status
map/map_client/CMakeFiles/map_client_v1.dir/build.make:799: recipe for target '/home/guanggang.bian/Private/Code/common/devel/lib/libmap_client_v1.so' failed
make[2]: *** [/home/guanggang.bian/Private/Code/common/devel/lib/libmap_client_v1.so] Error 1
CMakeFiles/Makefile2:5014: recipe for target 'map/map_client/CMakeFiles/map_client_v1.dir/all' failed
make[1]: *** [map/map_client/CMakeFiles/map_client_v1.dir/all] Error 2
/usr/bin/ld: //usr/local/lib/libpqxx.a(connection.o): relocation R_X86_64_PC32 against symbol `_ZTVN4pqxx16connectionpolicyE' can not be used when making a shared object; recompile with -fPIC
/usr/bin/ld: final link failed: Bad value
collect2: error: ld returned 1 exit status
map/hdmap/CMakeFiles/map_hdmap.dir/build.make:747: recipe for target '/home/guanggang.bian/Private/Code/common/devel/lib/libmap_hdmap.so' failed
make[2]: *** [/home/guanggang.bian/Private/Code/common/devel/lib/libmap_hdmap.so] Error 1
CMakeFiles/Makefile2:4623: recipe for target 'map/hdmap/CMakeFiles/map_hdmap.dir/all' failed
make[1]: *** [map/hdmap/CMakeFiles/map_hdmap.dir/all] Error 2
Makefile:140: recipe for target 'all' failed
make: *** [all] Error 2
Invoking "make install . -j24" failed
```

### 解决方法
该问题是因为 使用 C++11 编译 libqpxx 部分功能被废弃的原因

可以重新编译，指定 存放的 地点为 /usr/local

参考如下
```
wget http://pqxx.org/download/software/libpqxx/libpqxx-4.0.1.tar.gz
tar -xzvf libpqxx-4.0.1.tar.gz
cd libpqxx-4.0.1
./configure --prefix=/usr/local --enable-shared
make clean
make
sudo make install
```