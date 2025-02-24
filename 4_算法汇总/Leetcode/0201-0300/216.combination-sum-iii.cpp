/*
 * @lc app=leetcode id=216 lang=cpp
 *
 * [216] Combination Sum III
 *
 * https://leetcode.com/problems/combination-sum-iii/description/
 *
 * algorithms
 * Medium (69.60%)
 * Likes:    6238
 * Dislikes: 115
 * Total Accepted:    623.2K
 * Total Submissions: 873.9K
 * Testcase Example:  '3\n7'
 *
 * Find all valid combinations of k numbers that sum up to n such that the
 * following conditions are true:
 * 
 * 
 * Only numbers 1 through 9 are used.
 * Each number is used at most once.
 * 
 * 
 * Return a list of all possible valid combinations. The list must not contain
 * the same combination twice, and the combinations may be returned in any
 * order.
 * 
 * 
 * Example 1:
 * 
 * Input: k = 3, n = 7
 * Output: [[1,2,4]]
 * Explanation:
 * 1 + 2 + 4 = 7
 * There are no other valid combinations.
 * 
 * Example 2:
 * 
 * Input: k = 3, n = 9
 * Output: [[1,2,6],[1,3,5],[2,3,4]]
 * Explanation:
 * 1 + 2 + 6 = 9
 * 1 + 3 + 5 = 9
 * 2 + 3 + 4 = 9
 * There are no other valid combinations.
 * 
 * 
 * Example 3:
 * 
 * Input: k = 4, n = 1
 * Output: []
 * Explanation: There are no valid combinations.
 * Using 4 different numbers in the range [1,9], the smallest sum we can get is
 * 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 2 <= k <= 9
 * 1 <= n <= 60
 * 
 * 
 */
#include <vector>
#include <iostream>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<int> dp;
        dp.reserve(k);
        vector<vector<int>> res;
        dfs(k, n, dp, res);
        return res;
    }

    void dfs(int k, int n, vector<int>& dp, vector<vector<int>>& res) {
        if (k == 1) {
            if (dp.size() == 0) {
                res.emplace_back(vector<int>{n});
            } else if (n > dp.back() && n <= 9) {
                dp.emplace_back(n);
                res.emplace_back(dp);
                dp.pop_back();
            }
            return;
        }
        int start = 1;
        if (dp.size() != 0) {
            start = dp.back() + 1;
        }
        for (int i = start; i < 10; i++) {
            int tmp_n = n - i;
            if (tmp_n <= i) break;
            dp.emplace_back(i);
            dfs(k-1, tmp_n, dp, res);
            dp.pop_back();
        }
    }
};
// @lc code=end

int main() {
    Solution s = Solution();
    vector<vector<int>> res = s.combinationSum3(3, 9);
    for (auto set_num : res) {
        for (auto num : set_num) {
            cout << num << " ";            
        }
        cout << endl;
    }
    return 0;
}


// 18/18 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 45.25 % of cpp submissions (8.9 MB)
