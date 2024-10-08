/*
 * @lc app=leetcode id=24 lang=cpp
 *
 * [24] Swap Nodes in Pairs
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
    ListNode* swapPairs(ListNode* head) {
        ListNode res_head_node;
        ListNode* res = &res_head_node;
        res->next = head;
        while (res->next != nullptr) {
            ListNode* first = res->next;
            if (first->next == nullptr) break;
            ListNode* second = first->next;
            res->next = second;
            first->next = second->next;
            second->next = first;
            res = res->next->next;
        }
        return res_head_node.next;
    }
};
// @lc code=end

// 55/55 cases passed (5 ms)
// Your runtime beats 14.86 % of cpp submissions
// Your memory usage beats 82.52 % of cpp submissions (9.4 MB)