/*
 * @lc app=leetcode id=23 lang=cpp
 *
 * [23] Merge k Sorted Lists
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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        if (lists.size() == 1) return lists[0];
        if (lists.size() == 2) return mergeTwoLists(lists[0], lists[1]);
        int mid = lists.size() / 2;
        vector<ListNode*> part1(lists.begin(), lists.begin()+mid);
        vector<ListNode*> part2(lists.begin()+mid, lists.end());
        ListNode* mergepart1 = mergeKLists(part1);
        ListNode* mergepart2 = mergeKLists(part2);
        return mergeTwoLists(mergepart1, mergepart2);
    }

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

// 134/134 cases passed (16 ms)
// Your runtime beats 64.52 % of cpp submissions
// Your memory usage beats 6.4 % of cpp submissions (28.6 MB)