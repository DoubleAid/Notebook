/*
 * @lc app=leetcode id=403 lang=cpp
 *
 * [403] Frog Jump
 *
 * https://leetcode.com/problems/frog-jump/description/
 *
 * algorithms
 * Hard (46.03%)
 * Likes:    5629
 * Dislikes: 255
 * Total Accepted:    278.4K
 * Total Submissions: 602.3K
 * Testcase Example:  '[0,1,3,5,6,8,12,17]'
 *
 * A frog is crossing a river. The river is divided into some number of units,
 * and at each unit, there may or may not exist a stone. The frog can jump on a
 * stone, but it must not jump into the water.
 * 
 * Given a list of stones positions (in units) in sorted ascending order,
 * determine if the frog can cross the river by landing on the last stone.
 * Initially, the frog is on the first stone and assumes the first jump must be
 * 1 unit.
 * 
 * If the frog's last jump was k units, its next jump must be either k - 1, k,
 * or k + 1 units. The frog can only jump in the forward direction.
 * 
 * 
 * Example 1:
 * 
 * Input: stones = [0,1,3,5,6,8,12,17]
 * Output: true
 * Explanation: The frog can jump to the last stone by jumping 1 unit to the
 * 2nd stone, then 2 units to the 3rd stone, then 2 units to the 4th stone,
 * then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the
 * 8th stone.
 * 
 * 
 * Example 2:
 * 
 * Input: stones = [0,1,2,3,4,8,9,11]
 * Output: false
 * Explanation: There is no way to jump to the last stone as the gap between
 * the 5th and 6th stone is too large.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 2 <= stones.length <= 2000
 * 0 <= stones[i] <= 2^31 - 1
 * stones[0] == 0
 * stones is sorted in a strictly increasing order.
 * 
 * 
 */
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// @lc code=start
class Solution {
public:
    bool canCross(vector<int>& stones) {
        if (stones[1] - stones[0] != 1) return false; 

        unordered_map<int, unordered_set<int>> steps;
        for (int stone : stones) {
            steps[stone] = unordered_set<int>();
        }

        steps[stones[1]].insert(1);

        for (int stone : stones) {
            for (int step : steps[stone]) {
                if (step - 1 > 0 && steps.count(stone + step - 1) != 0) {
                    steps[stone+step-1].insert(step-1);
                }
                if (steps.count(stone+step) != 0) {
                    steps[stone + step].insert(step);
                }
                if (steps.count(stone+step+1) != 0) {
                    steps[stone + step + 1].insert(step+1);
                }
            }
        }

        return !steps[stones.back()].empty();
    }
};
// @lc code=end

int main() {
    Solution s;
    vector<int> num = {0, 1, 3, 5, 6, 8, 12, 17};
    s.canCross(num);
    return 0;
}

55/55 cases passed (123 ms)
Your runtime beats 40.74 % of cpp submissions
Your memory usage beats 54.41 % of cpp submissions (44.3 MB)