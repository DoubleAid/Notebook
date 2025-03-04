#include <thread>
#include <mutex>
#include <shared_mutex>
#include <chrono>

std::shared_timed_mutex mtx;
int shared_data1 = 0;

void read_data() {
    std::shared_lock<std::shared_timed_mutex> lock(mtx, std::defer_lock);
    if (lock.try_lock_for(std::chrono::seconds(1))) {
        std::cout << "Read Data: " << shared_data1 << std::endl;
    } else {
        std::cout << "Failed to acquire shared lock\n" << std::endl;
    }
}

void write_data() {
    std::unique_lock<std::shared_timed_mutex> lock(mtx, std::defer_lock);
    if (lock.try_lock_for(std::chrono::seconds(1))) {
        shared_data1++;
        std::cout << "Write Data: " << shared_data1 << std::endl;
    } else {
        std::cout << "Fail to acquire unique lock" << std::endl;
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(read_data);
    }

    for (int i = 0; i < 5; i++) {
        threads.emplace_back(write_data);
    }

    for (auto& t : threads) {
        t.join();
    }
    return 0;
}