/*
 * @lc app=leetcode id=92 lang=cpp
 *
 * [92] Reverse Linked List II
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
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode m(0, head);
        ListNode *start = &m, *end = nullptr;
        int count = 1;
        while (count < left) {
            count++;
            start = start->next;
        }
        end = start;
        while (count <= right) {
            count++;
            end=end->next;
        }
        ListNode *temp_rear = end->next, *temp_start = start->next;
        while (temp_start != end) {
            ListNode *temp = temp_start;
            temp_start = temp_start->next;
            temp->next = temp_rear;
            temp_rear = temp;
        }
        start->next = end;
        end->next = temp_rear;
        return m.next;
    }
};
// @lc code=end

// 44/44 cases passed (2 ms)
// Your runtime beats 50.28 % of cpp submissions
// Your memory usage beats 53.97 % of cpp submissions (9.4 MB)