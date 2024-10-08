/*
 * @lc app=leetcode id=25 lang=cpp
 *
 * [25] Reverse Nodes in k-Group
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode res(0, head);
        ListNode* next_head = &res;
        while (next_head->next != nullptr) {
            ListNode* tmp_record = next_head->next;
            next_head->next = reverseFirstGroup(next_head->next, k);
            if (tmp_record == next_head->next) break;
            next_head = tmp_record;
        }
        return res.next;
    }

    ListNode* reverseFirstGroup(ListNode* head, int k) {
        ListNode tmp_node;
        ListNode* tmp_head = &tmp_node;
        ListNode* p = tmp_head;
        ListNode* end = nullptr;
        ListNode* ctn = head;
        int count = 0;
        while (ctn != nullptr) {
            count++;
            ctn = ctn->next;
        }
        if (count < k) return head;
        while (k > 0) {
            ListNode* m = head;
            head = head->next;
            m->next = p->next;
            p->next = m;
            if (end == nullptr) end = m;
            k--;
        }
        end->next = head;
        return tmp_head->next;
    }
};
// @lc code=end

// 62/62 cases passed (11 ms)
// Your runtime beats 44.13 % of cpp submissions
// Your memory usage beats 96.36 % of cpp submissions (14.9 MB)