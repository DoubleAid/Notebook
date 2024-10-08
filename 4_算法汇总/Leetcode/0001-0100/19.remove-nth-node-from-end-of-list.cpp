/*
 * @lc app=leetcode id=19 lang=cpp
 *
 * [19] Remove Nth Node From End of List
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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* p = new ListNode(0, head);
        ListNode* q = p;
        ListNode* res = p;
        for (int i = 0; i < n; i++) {
            q = q -> next;
        }
        while (q->next != nullptr) {
            q = q -> next;
            p = p -> next;
        }
        p->next = p->next->next;
        return res->next;
    }
};
// @lc code=end

// 208/208 cases passed (3 ms)
// Your runtime beats 63.64 % of cpp submissions
// Your memory usage beats 18.78 % of cpp submissions (13.4 MB)