# 698 [M] Partition to K Equal Sum Subsets
Given an integer array nums and an integer k, return true if it is possible to divide this array into k non-empty subsets whose sums are all equal.

#### Example 1:
```
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: It is possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
```
#### Example 2:
```
Input: nums = [1,2,3,4], k = 3
Output: false
```
#### Constraints:
+ 1 <= k <= nums.length <= 16
+ 1 <= nums[i] <= 104
+ The frequency of each element is in the range [1, 4]

#### 思路和解决方法
第一种我看了网上的一种贪心算法， 申请 k 个空间， 依次把 数 放进空间内， 主要是找第一个可以放这个数的空间， 如果放不下， 就把这个数放到下一个空间内， 知道所有的数都放完， 代码如下：
```cpp
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int total = accumulate(nums.begin(), nums.end(), 0);
        if (total % k != 0) return false;
        int value = int(total / k);
        sort(nums.begin(), nums.end());
        vector<int> group(k);
        return add(nums, value, group, (int)nums.size()-1);
    }
    
    bool add(vector<int>& nums, int target, vector<int>& group, int idx) {
        if (idx == -1) {
            for (int value : group) {
                if (value != target) return false;
            }
            return true;
        }
        int num = nums[idx];
        for (int i = 0; i < group.size(); i++) {
            if (group[i] + num > target) continue;
            group[i] += num;
            if (add(nums, target, group, idx-1)) return true;
            group[i] -= num;
        }
        return false;
    }
};
```

提交时，卡在一个测试用例， 测试用例超时
```
[3,9,4,5,8,8,7,9,3,6,2,10,10,4,10,2]
10
```
这个测试用例 和为 100， target = 10， 当需要 放入 第一个 9 时， group 中的前三个都是 10， 后面都是 0， 放入第四个空间， 没有配合的1， `add(nums, target, group, idx-1)` 返回 false， 接着 挪到下一个空间，重复测试， 直到第十个空间， 因此需要检测放在第四个空间不合适， 将 9 拿出空间后， 空间如果为空， 就需要 返回 false 了
改正如下：
```cpp
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int total = accumulate(nums.begin(), nums.end(), 0);
        if (total % k != 0) return false;
        int value = int(total / k);
        sort(nums.begin(), nums.end());
        vector<int> group(k);
        return add(nums, value, group, (int)nums.size()-1);
    }
    
    bool add(vector<int>& nums, int target, vector<int>& group, int idx) {
        if (idx == -1) {
            for (int value : group) {
                if (value != target) return false;
            }
            return true;
        }
        int num = nums[idx];
        for (int i = 0; i < group.size(); i++) {
            if (group[i] + num > target) continue;
            group[i] += num;
            if (add(nums, target, group, idx-1)) return true;
            group[i] -= num;
            if (group[i] == 0) break;
        }
        return false;
    }
};
```

另外还有一种是 递归方法， 设置一个标记为， 访问了标记为 true， 未访问标记为 false， 在数组 nums 中查找 `target - curValue`, 格式和上面比较类似

```cpp
class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % k != 0) return false;
        vector<bool> visited(nums.size());
        return helper(nums, k, sum / k, 0, 0, visited);
    }
    bool helper(vector<int>& nums, int k, int target, int start, int curSum, vector<bool>& visited) {
        if (k == 1) return true;
        if (curSum == target) return helper(nums, k - 1, target, 0, 0, visited);
        for (int i = start; i < nums.size(); ++i) {
            if (visited[i]) continue;
            visited[i] = true;
            if (helper(nums, k, target, i + 1, curSum + nums[i], visited)) return true;
            visited[i] = false;
        }
        return false;
    }
};
```