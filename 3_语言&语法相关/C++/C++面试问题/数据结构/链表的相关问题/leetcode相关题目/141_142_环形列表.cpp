#include <iostream>
#include <unordered_map>

using namespace std;

template <class T>
class MyNode {
public:
    T data;
    MyNode* next;
    MyNode(): next(nullptr) {
        data = T();
    }
    MyNode(T x) : data(x), next(nullptr) {}
    MyNode(T x, MyNode* next_) : data(x), next(next_) {}
    ~MyNode() = default;
};

class ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};

// 使用 hashmap 计算, 速度比较慢
bool hasCycle(ListNode* head) {
    unordered_map<ListNode*, int> m;
    ListNode* p = head;
    while(p) {
        if(m.count(p) > 0) return true;
        m[p] = 0;
        p = p->next;
    }
    return false;
}

// 快慢指针
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head) return false;
        ListNode* p = head, *q = head;
        while(q->next && q->next->next) {
            p=p->next;
            q=q->next->next;
            if(p==q) return true;
        }
        return false;
    }
};

// 获取 环开始的位置
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(!head) return NULL;
        ListNode *fast = head, *slow = head;
        while(fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
            if(fast==slow) break;
        }
        if(!fast->next || !fast->next->next) return NULL;
        fast = head;
        while(fast != slow) {
            fast = fast->next;
            slow = slow->next;
        }
        return fast;
    }
};