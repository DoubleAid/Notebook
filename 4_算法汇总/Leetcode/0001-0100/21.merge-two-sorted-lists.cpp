/*
 * @lc app=leetcode id=21 lang=cpp
 *
 * [21] Merge Two Sorted Lists
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
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* res = new ListNode();
        ListNode* res_end = res;
        while (list1 != nullptr && list2 != nullptr) {
            if (list1->val > list2->val) {
                res_end->next = list2;
                res_end = list2;
                list2 = list2->next;
            } else {
                res_end->next = list1;
                res_end = list1;
                list1 = list1->next;
            }
        }
        if (list1 != nullptr) list2 = list1;
        res_end->next = list2;
        return res->next;
    }
};
// @lc code=end

// 208/208 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 74.7 % of cpp submissions (18.3 MB)
