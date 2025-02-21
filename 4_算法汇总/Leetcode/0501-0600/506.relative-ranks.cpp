/*
 * @lc app=leetcode id=506 lang=cpp
 *
 * [506] Relative Ranks
 *
 * https://leetcode.com/problems/relative-ranks/description/
 *
 * algorithms
 * Easy (63.78%)
 * Likes:    1987
 * Dislikes: 134
 * Total Accepted:    334.8K
 * Total Submissions: 459.5K
 * Testcase Example:  '[5,4,3,2,1]'
 *
 * You are given an integer array score of size n, where score[i] is the score
 * of the i^th athlete in a competition. All the scores are guaranteed to be
 * unique.
 * 
 * The athletes are placed based on their scores, where the 1^st place athlete
 * has the highest score, the 2^nd place athlete has the 2^nd highest score,
 * and so on. The placement of each athlete determines their rank:
 * 
 * 
 * The 1^st place athlete's rank is "Gold Medal".
 * The 2^nd place athlete's rank is "Silver Medal".
 * The 3^rd place athlete's rank is "Bronze Medal".
 * For the 4^th place to the n^th place athlete, their rank is their placement
 * number (i.e., the x^th place athlete's rank is "x").
 * 
 * 
 * Return an array answer of size n where answer[i] is the rank of the i^th
 * athlete.
 * 
 * 
 * Example 1:
 * 
 * Input: score = [5,4,3,2,1]
 * Output: ["Gold Medal","Silver Medal","Bronze Medal","4","5"]
 * Explanation: The placements are [1^st, 2^nd, 3^rd, 4^th, 5^th].
 * 
 * Example 2:
 * 
 * Input: score = [10,3,8,9,4]
 * Output: ["Gold Medal","5","Bronze Medal","Silver Medal","4"]
 * Explanation: The placements are [1^st, 5^th, 3^rd, 2^nd, 4^th].
 * 
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == score.length
 * 1 <= n <= 10^4
 * 0 <= score[i] <= 10^6
 * All the values in score are unique.
 * 
 * 
 */

#include <vector>
#include <string>
#include <algorithm>
#include <map>

using namespace std;

// @lc code=start
class Solution {
public:
    vector<string> findRelativeRanks(vector<int>& score) {
        map<int, string> rank_map;
        int rank = 1;
        vector<int> copy_score(score);
        std::sort(score.begin(), score.end(), [](const int& a, const int& b) {
            return a > b;
        });
        for (auto s : score) {
            if (rank == 1) {
                rank_map.emplace(s, "Gold Medal");
            } else if (rank == 2) {
                rank_map.emplace(s, "Silver Medal");
            } else if (rank == 3) {
                rank_map.emplace(s, "Bronze Medal");
            } else {
                rank_map.emplace(s, to_string(rank));
            }
            rank++;
        }
        vector<string> res;
        res.reserve(score.size());
        for (auto score : copy_score) {
            res.emplace_back(rank_map[score]);
        }
        return res;
    }
};
// @lc code=end

// 18/18 cases passed (7 ms)
// Your runtime beats 48.18 % of cpp submissions
// Your memory usage beats 35.51 % of cpp submissions (17.3 MB)