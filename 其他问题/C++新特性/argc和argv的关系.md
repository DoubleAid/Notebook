对于 main 函数，如果不需要读取任何命令时，是不需要参数的，但如果需要参数读取时，需要加上 argc 和 argv 两个参数
以一下代码为例
```cpp
#include <iostream>
using namespace std;
int main(int argc, char** argv) {
    cout << tostring(argc) << endl;
    for (int i=0; i<argc; i++) {
        cout << argv[i] << endl;
    }
    return 0;
}

```