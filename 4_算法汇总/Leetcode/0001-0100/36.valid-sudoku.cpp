/*
 * @lc app=leetcode id=36 lang=cpp
 *
 * [36] Valid Sudoku
 */

#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            vector<bool> flag(9, false);
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '1';
                    if (!flag[num]) {
                        flag[num] = true;
                    } else {
                        return false;
                    }
                }
            }
        }
        for (int j = 0; j < 9; j++) {
            vector<bool> flag(9, false);
            for (int i = 0; i < 9; i++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '1';
                    if (!flag[num]) {
                        flag[num] = true;
                    } else {
                        return false;
                    }
                }
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                vector<bool> flag(9, false);
                for (int m = i*3; m < i*3+3; m++) {
                    for (int n = j*3; n < j*3+3; n++) {
                        if (board[m][n] != '.') {
                            int num = board[m][n] - '1';
                            if (!flag[num]) {
                                flag[num] = true;
                            } else {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};
// @lc code=end

// 507/507 cases passed (7 ms)
// Your runtime beats 98.27 % of cpp submissions
// Your memory usage beats 72.71 % of cpp submissions (22.7 MB)