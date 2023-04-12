## vector 
+ 使用连续线性空间
+ 支持高效的随机访问和在尾端插入和删除的操作
+ 与数组类似，与数组的区别是动态的内存空间扩展

STL内部实现时，首先分配一个非常大的内存空间预备进行存储，即capacity（）函数返回的大小，
当超过此分配的空间时再整体重新放分配一块内存存储（ VS6.0是两倍，VS2005是1.5倍），
所以 这给人以vector可以不指定vector即一个连续内存的大小的感觉。通常此默认的内存分配能
完成大部分情况下的存储。

## list
+ 使用非连续存储， 内部实现是一个双向链表，
+ 支持高效的随机插入和删除操作， 随机访问效率较低， 内存占用较大

## deque
+ 二维列表， 一维存储第二维向量的指针，第二维是等长的
+ 适合在头尾添加和删除
+ 只有需要在首端进行插入/删除操作的时候，还要兼顾随机访问效率，才选择deque，
否则都选择vector。
+ 若既需要随机插入/删除，又需要随机访问，则需要在vector与list间做个折中 deque。

## map
+ map的底层是用红黑树实现的， 查找的时间复杂度是 log(n)
+ map 的默认key值排序（int）使用由小到大排序，string 默认按ascll码排序，短的优先
+ 可以指定排序方式
```c++
// 第一种
map<string, int, greater<string>> s;
map<string, int, greater<string>>::iterator it;

// 第二种 重载运算符
class sdu {
    int x, y, z;
    bool operator<(const sdu& o) const {
        return x < o.x || y < o.y;
    }
};

map<sdu, string> s;
...
```



## capacity 和 size
capacity 是指内存重新分配之前可以容纳的最大数据量， size 是当前容器实际占用的大小，
resize()
reserve()