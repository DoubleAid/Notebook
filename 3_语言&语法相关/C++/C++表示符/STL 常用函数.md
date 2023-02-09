### std::back_inserter
返回已存在数据的最有一个 迭代器
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main()
{
    std::vector<int> v1{ 1, 2, 3, 4, 5, 6};
    std::vector<int> v2{ 10, 20, 30, 40, 50, 60};

    std::copy(v2.begin(), v2.end(), std::back_inserter(v1));

    std::cout << "v1:   ";
    for (int i : v1) {
        std::cout << i << "\t";
    }
    std::cout << std::endl;

    return 0;
}
```

**与end()的不同**
首先end()返回的迭代器是不支持解引用的，要想修改最后一个元素使用v.end()-1，强行对end()解引用会报异常。

然后*(v.end()-1) = 10;是修改最后一个元素内容，而*(std::back_inserter(v2)) = 10;是在末尾新增一个元素。

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main()
{
    std::vector<int> v1{ 1, 2, 3, 4, 5, 6};
    std::vector<int> v2{ 1, 2, 3, 4, 5, 6 };

    *(v1.end()-1) = 10;
    *(std::back_inserter(v2)) = 10;

    std::cout << "v1: ";
    for (int i : v1) {
        std::cout << i << "\t";
    }
    std::cout << std::endl << std::endl;

    std::cout << "v2: ";
    for (int i : v2) {
        std::cout << i << "\t";
    }
    std::cout << std::endl << std::endl;

    system("pause");
    return 0;
}
```
