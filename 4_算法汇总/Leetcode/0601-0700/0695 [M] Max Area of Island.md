# 695 [M] Max Area of Island
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
The area of an island is the number of cells with a value 1 in the island.
Return the maximum area of an island in grid. If there is no island, return 0.

#### Example 1:
```
Input: grid = 
[
    [0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,1,0,0],
    [0,1,0,0,1,1,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0]
]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
```
#### Example 2:
```
Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0
```
#### Constraints:
+ m == grid.length
+ n == grid[i].length
+ 1 <= m, n <= 50
+ grid[i][j] is either 0 or 1.

#### 思路
这就是一个遍历的问题， 按行所有海域， 当发现有海岛时， 就测量海岛的大小， 并把海岛变成大海，
测量海岛的大小可以使用 BFS 或者 DFS
```cpp
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();
        int ans = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    std::queue<std::pair<int, int>> queue;
                    queue.emplace(i, j);
                    int count = 0;
                    while (!queue.empty()) {
                        std::pair<int, int> pos = queue.front();
                        queue.pop();
                        if (grid[pos.first][pos.second] == 0) continue;
                        grid[pos.first][pos.second] = 0;
                        count++;
                        if (pos.first > 0 && grid[pos.first-1][pos.second] == 1) queue.emplace(pos.first-1, pos.second);
                        if (pos.first + 1 < row && grid[pos.first+1][pos.second] == 1) queue.emplace(pos.first+1, pos.second);
                        if (pos.second > 0 && grid[pos.first][pos.second-1] == 1) queue.emplace(pos.first, pos.second-1);
                        if (pos.second + 1 < col && grid[pos.first][pos.second+1] == 1) queue.emplace(pos.first, pos.second+1);
                    }
                    ans = std::max(ans, count);
                }
            }
        }
        return ans;
    }
};
```