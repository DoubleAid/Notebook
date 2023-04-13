#include <mutex>
#include <iostream>

using namespace std;

class Task {
private:
    static Task* instance;
    static mutex mutex_; // 线程安全
    Task() {}
    ~Task() {}
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
public:
    static Task* getInstance() {
        if(!instance) {
            lock_guard<mutex> lock{mutex_};
            if (!instance) instance = new Task();
        }
        return instance;
    } 
};