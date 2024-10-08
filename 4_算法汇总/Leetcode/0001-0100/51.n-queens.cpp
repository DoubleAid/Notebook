/*
 * @lc app=leetcode id=51 lang=cpp
 *
 * [51] N-Queens
 *
 * https://leetcode.com/problems/n-queens/description/
 *
 * algorithms
 * Hard (68.09%)
 * Likes:    12181
 * Dislikes: 280
 * Total Accepted:    724.3K
 * Total Submissions: 1.1M
 * Testcase Example:  '4'
 *
 * The n-queens puzzle is the problem of placing n queens on an n x n
 * chessboard such that no two queens attack each other.
 * 
 * Given an integer n, return all distinct solutions to the n-queens puzzle.
 * You may return the answer in any order.
 * 
 * Each solution contains a distinct board configuration of the n-queens'
 * placement, where 'Q' and '.' both indicate a queen and an empty space,
 * respectively.
 * 
 * 
 * Example 1:
 * 
 * Input: n = 4
 * Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
 * Explanation: There exist two distinct solutions to the 4-queens puzzle as
 * shown above
 * 
 * 
 * Example 2:
 * 
 * Input: n = 1
 * Output: [["Q"]]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 9
 * 
 * 
 */
#include <vector>

using namespace std;
// @lc code=start
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> solutions;
        vector<string> board(n, string(n, '.')); // 初始化棋盘，所有位置为空
        vector<int> cols(n, 0);  // 列的标记
        vector<int> diag1(2 * n - 1, 0); // 主对角线标记
        vector<int> diag2(2 * n - 1, 0); // 副对角线标记
        backtrack(0, n, cols, diag1, diag2, board, solutions);
        return solutions;
    }

    void backtrack(int row, int n, vector<int>& cols, vector<int>& diag1, vector<int>& diag2,
                   vector<string>& board, vector<vector<string>>& solutions) {
        if (row == n) {
            // 找到一个解决方案
            solutions.push_back(board);
            return;
        }

        for (int col = 0; col < n; col++) {
            // 检查是否可以放置皇后
            if (cols[col] || diag1[row + col] || diag2[row - col + n - 1]) {
                continue; // 如果不行，尝试下一列
            }

            // 放置皇后
            board[row][col] = 'Q';
            cols[col] = 1;
            diag1[row + col] = 1;
            diag2[row - col + n - 1] = 1;

            // 递归到下一行
            backtrack(row + 1, n, cols, diag1, diag2, board, solutions);

            // 撤销操作，尝试下一个位置
            board[row][col] = '.';
            cols[col] = 0;
            diag1[row + col] = 0;
            diag2[row - col + n - 1] = 0;
        }
    }
};
// @lc code=end

// 9/9 cases passed (0 ms)
// Your runtime beats 100 % of cpp submissions
// Your memory usage beats 71.09 % of cpp submissions (9.3 MB)