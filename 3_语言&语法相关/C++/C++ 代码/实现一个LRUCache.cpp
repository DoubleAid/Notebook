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
        // 
        lru_.splice(lru_.begin(), lru_, it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second->second = value;
            lru_.splice(lru_.begin(), lru_, it->second);
        } else {
            if (lru_.size() == capacity_) {
                auto last = lru_.back();
                cache_.erase(last.first);
                lru_.pop_back();
            }
            lru_.emplace_front(key, value);
            cache_[key] = lru_.begin();
        }
    }
private:
    int capacity_;
    std::list<std::pair<int, int>> lru_;
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> cache_;
};

int main() {
    LRUCache lruCache(2);

    lruCache.put(1, 2);
    lruCache.put(2, 4);
    
    std::cout << lruCache.get(1) << std::endl;

    lruCache.put(3, 6);

    std::cout << lruCache.get(2) << std::endl;

    lruCache.put(4, 8);

    std::cout << lruCache.get(1) << std::endl;
    std::cout << lruCache.get(3) << std::endl;
    std::cout << lruCache.get(4) << std::endl;

    return 0;
}