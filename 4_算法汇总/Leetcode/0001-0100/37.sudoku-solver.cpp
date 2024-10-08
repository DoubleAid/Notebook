/*
 * @lc app=leetcode id=37 lang=cpp
 *
 * [37] Sudoku Solver
 */

// @lc code=start
class Solution {
public:
    bool rowValid(vector<vector<char>>& board, int i) {
        vector<bool> flag(9, false);
        for (int m = 0; m < 9; m++) {
            if (board[i][m] != '.') {
                int index = board[i][m] - '1';
                if (!flag[index]) {
                    flag[index] = true;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    bool colValid(vector<vector<char>>& board, int j) {
        vector<bool> flag(9, false);
        for (int m = 0; m < 9; m ++) {
            if (board[m][j] != '.') {
                int index = board[m][j] - '1';
                if (!flag[index]) {
                    flag[index] = true;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    bool squareValid(vector<vector<char>>& board, int i, int j) {
        int s_i = i / 3, s_j = j / 3;
        vector<bool> flag(9, false);
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                int index_i = s_i * 3 + m;
                int index_j = s_j * 3 + n;
                if (board[index_i][index_j] != '.') {
                    int index = board[index_i][index_j] - '1';
                    if (!flag[index]) {
                        flag[index] = true;
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool solve(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (int m = 1; m < 10; m++) {
                        board[i][j] = '0' + m;
                        if (rowValid(board, i) &&
                            colValid(board, j) &&
                            squareValid(board, i, j)) {
                            if (solve(board)) return true;
                        }
                    }
                    board[i][j] = '.';
                    return false;
                }
            }
        }
        return true;
    }

    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
    }
};
// @lc code=end

// 6/6 cases passed (241 ms)
// Your runtime beats 5.02 % of cpp submissions
// Your memory usage beats 5 % of cpp submissions (37.5 MB)