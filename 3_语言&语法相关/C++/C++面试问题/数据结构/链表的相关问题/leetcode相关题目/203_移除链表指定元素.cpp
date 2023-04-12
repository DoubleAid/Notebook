//
// Created by guanggang.bian on 2023/4/6.
//


// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution1 {
public:
    ListNode* removeElements(ListNode* head, int val) {
        while(head != nullptr && head->val == val) {
            head = head->next;
        }
        if (head == nullptr) return head;
        ListNode* p = head;
        ListNode* q = p->next;
        while (q != nullptr) {
            if (q->val == val) {
                p->next = q->next;
                q = q->next;
            } else {
                p = q;
                q = q->next;
            }
        }
        return head;
    }
};

class Solution2 {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* fake_head = new ListNode(-1), *pre = fake_head;
        pre->next = head;
        while (pre->next) {
            if (pre->next->val == val) {
                ListNode* tmp = pre->next;
                pre->next = pre->next->next;
                delete tmp;
            }
            else {
                pre = pre->next;
            }
        }
        pre = fake_head->next;
        delete fake_head;
        return pre;
    }
};

// 计算环的开始的位置
