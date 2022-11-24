# 337. House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

#### Example 1:
```
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```
#### Example 2:
```
Input: root = [3,4,5,1,3,null,1]
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
```
#### Constraints:
+ The number of nodes in the tree is in the range [1, 104].
+ 0 <= Node.val <= 104

#### 思路
calculate函数返回当前结点为根结点的最大 rob 的钱数，里面的两个参数l和r表示分别从左子结点和右子结点开始 rob，分别能获得的最大钱数。在递归函数里面，如果当前结点不存在，直接返回0。否则对左右子结点分别调用递归函数，得到l和r。另外还得到四个变量，ll和lr表示左子结点的左右子结点的最大 rob 钱数，rl 和 rr 表示右子结点的最大 rob 钱数。那么最后返回的值其实是两部分的值比较，其中一部分的值是当前的结点值加上 ll, lr, rl, 和 rr 这四个值，这不难理解，因为抢了当前的房屋，则左右两个子结点就不能再抢了，但是再下一层的四个子结点都是可以抢的；另一部分是不抢当前房屋，而是抢其左右两个子结点，即 l+r 的值，返回两个部分的值中的较大值即可
```cpp
class Solution {
public:
    int rob(TreeNode* root) {
        int l = 0, r = 0;
        return calculate(root, l, r);
    }

    int calculate(TreeNode* root, int& l, int& r) {
        if (!root) return 0;
        int ll = 0, lr = 0, rl = 0, rr = 0;
        l = calculate(root->left, ll, lr);
        r = calculate(root->right, rl, rr);
        return max(root->val+ll+lr+rr+rl, l+r);
    }
};
```
