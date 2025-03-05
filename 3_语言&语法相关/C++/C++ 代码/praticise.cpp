#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <thread>

std::shared_timed_mutex mx;

int shared_data;

void reader(int thread_id) {
    while (true) {
        std::shared_lock<std::timed_mutex> lock(mx, std::defer_lock);
        if (lock.try_lock_for(std::chrono::seconds(1))) {
            std::cout << "Reader " << thread_id << " read shared data " << shared_data << std::endl;
        } else {
            std::cout << "Reader " << thread_id << " cant not read shared data" << std::endl;
        }
    }
}

void writer(int thread_id) {
    while (true) {
        std::shared_lock<std::timed_mutex> lock{mx};
        if (lock.try_lock_for(std::chrono::seconds(1))) {
            shared_data++;
            std::cout << "Writer " << thread_id << " write shared data " << shared_data << std::endl;
        } else {
            std::cout << "Writer " << thread_id << " cant not write shared data" << std::endl;
        }
    }
}

int main() {
    for (int i = 0; i < 10; i++) {
        std::thread(reader, i);
    }
    for (int i = 0; i < 3; i++) {
        std::thread(writer, i);
    }
    while (1) {
        sleep(1);
    }
}