//
// Created by guanggang.bian on 2023/4/6.
//
// 206
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* new_head = nullptr, *tmp;
        while(head) {
            tmp = head;
            head = head->next;
            tmp->next = new_head;
            new_head = tmp;
        }
        return new_head;
    }
};

// 92
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* pre_head = new ListNode(), *p = pre_head;
        pre_head->next = head;
        int i = 1;
        while(i < left) {p = p->next; i++;}
        ListNode* reverse_head = p->next, *reverse_tail = p->next, *head_tail = p;
        p = p->next;
        while(i <= right) {
            auto tmp = p;
            p = p->next;
            tmp->next = reverse_head;
            reverse_head = tmp;
            i++;
        }
        reverse_tail->next = p;
        head_tail->next = reverse_head;
        return pre_head->next;
    }
};