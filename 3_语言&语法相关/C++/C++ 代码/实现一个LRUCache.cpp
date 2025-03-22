// LRU是一个 key value 的缓存器，通常包含指定的长度，get 和 put 操作
// 当 长度不足时删除 最近最少使用的 key 值

#include <iostream>
#include <unordered_map>
#include <list>

class LRUCache {
public:
    LRUCache(int capacity) : capacity_(capacity) {}

    int get(int key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return -1;
        }

        lru_.splice()
    }
private:
    int capacity_;
    std::list<std::pair<int, int>> lru_;
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> cache_;
};