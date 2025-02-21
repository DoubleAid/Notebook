/*
 * @lc app=leetcode id=149 lang=cpp
 *
 * [149] Max Points on a Line
 *
 * https://leetcode.com/problems/max-points-on-a-line/description/
 *
 * algorithms
 * Hard (26.69%)
 * Likes:    4310
 * Dislikes: 534
 * Total Accepted:    444.6K
 * Total Submissions: 1.6M
 * Testcase Example:  '[[1,1],[2,2],[3,3]]'
 *
 * Given an array of points where points[i] = [xi, yi] represents a point on
 * the X-Y plane, return the maximum number of points that lie on the same
 * straight line.
 * 
 * 
 * Example 1:
 * 
 * Input: points = [[1,1],[2,2],[3,3]]
 * Output: 3
 * 
 * 
 * Example 2:
 * 
 * Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
 * Output: 4
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= points.length <= 300
 * points[i].length == 2
 * -10^4 <= xi, yi <= 10^4
 * All the points are unique.
 * 
 * 
 */

 #define debug 1
 #if debug
 #include <vector>
 #include <utility>
 #include <string>
 #include <unordered_map>
 #endif
 
 using namespace std;
 
 // @lc code=start
 
 class Solution {
 public:
     int maxPoints(vector<vector<int>>& points) {
         if (points.size() <= 2) return points.size();
         vector<int> master = points.back();
         points.pop_back();
         unordered_map<string, int> dp;
         int max_count = 0;
         for (auto p : points) {
             double dx = master[0] - p[0];
             double dy = master[1] - p[1];
             if (dx == 0) dy = 1;
             else if (dy == 0) dx = 1;
             else {
                 dy = dy / dx;
                 dx = 1;
             }
             string str = to_string(dx) + "_" + to_string(dy);
             if (dp.find(str) == dp.end()) {
                 dp.emplace(str, 2);
             } else {
                 dp[str] += 1;
             }
             if (dp[str] > max_count) max_count = dp[str];
         }
         return max(max_count, maxPoints(points));
     }
 };
 // @lc code=end

// 41/41 cases passed (125 ms)
// Your runtime beats 6.58 % of cpp submissions
// Your memory usage beats 5.02 % of cpp submissions (33.7 MB)
