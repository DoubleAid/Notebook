// 2373. Largest Local Values in a Matrix
// Easy
// Topics
// Companies
// Hint
// You are given an n x n integer matrix grid.

// Generate an integer matrix maxLocal of size (n - 2) x (n - 2) such that:

// maxLocal[i][j] is equal to the largest value of the 3 x 3 matrix in grid centered around row i + 1 and column j + 1.
// In other words, we want to find the largest value in every contiguous 3 x 3 matrix in grid.

// Return the generated matrix.

#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<vector<int>> largestLocal(vector<vector<int>>& grid) {
        int len = grid.size();
        vector<vector<int>> ret(len-2, vector<int>(len-2, 0));
        
    }
};

int main() {
    Solution s();
    s
}