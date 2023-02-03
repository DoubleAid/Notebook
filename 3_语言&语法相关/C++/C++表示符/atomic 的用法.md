std::atomic 用来保证原子操作

### 用下面三个例子进行说明
以下例子以100个线程一起执行，并且同时将全局变量 cnt 取出來 +1 计数，
但是由于多个执行线程同时执行存取 cnt 的关系会造成数据不正确。
来看看結果輸出会是怎么样吧！
```cpp
// g++ std-atomic.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>

using namespace std;

long cnt = 0;

void counter()
{
    for (int i = 0; i < 100000; i++) {
        cnt += 1;
    }
}

int main(int argc, char* argv[])
{
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();
    std::cout << "result: " << cnt << std::endl;
    return 0;
}
```
输出结果
```
result: 1866806
```
因为 数据从内存读取到寄存器上时， 还存在其他线程也会读取内存中的数据， 导致结果不对

### 添加 mutex 来解决问题
加 mutex 锁来保护临界区域是最常见的做法。可以保证同一时间内只有一个线程会存取 cnt
```cpp
// g++ std-atomic2.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

long cnt = 0;
std::mutex mtx;

void counter()
{
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        //std::cout << std::this_thread::get_id() << ": " << cnt << '\n';
        //mtx.lock();
        cnt += 1;
        //mtx.unlock();
    }
}

int main(int argc, char* argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "result: " << cnt << std::endl;
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    return 0;
}
```
输出：
```
result: 10000000
duration: 1426.77 ms
```
输出是正确的， 但是耗时比较长

### 使用 atomic 達到同樣效果，時間約少了7倍
接下來這裡介紹本篇重頭戲 atomic，
如果對象是 long 的話，可以用 std::atomic<long>，也可以用 std::atomic_long這個類別，
用 atomic 也可以達到同樣的效果，但所花費的時間有減少嗎？
來看看結果輸出會是怎樣吧！
```cpp
// g++ std-atomic3.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <atomic>

using namespace std;

//std::atomic<long> cnt(0);
std::atomic_long cnt(0);

void counter()
{
    for (int i = 0; i < 100000; i++) {
        cnt += 1;
    }
}

int main(int argc, char* argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "result: " << cnt << std::endl;
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    return 0;
}
```
运行结果：
```
result: 10000000
duration: 225.587 ms
```