# 697 [E] Degree of an Array
Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency of any one of its elements.

Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.

#### Example 1:
```
Input: nums = [1,2,2,3,1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
```
#### Example 2:
```
Input: nums = [1,2,2,3,1,4,2]
Output: 6
Explanation: 
The degree is 3 because the element 2 is repeated 3 times.
So [2,2,3,1,4,2] is the shortest subarray, therefore returning 6.
``` 

#### Constraints:
+ nums.length will be between 1 and 50,000.
+ nums[i] will be an integer between 0 and 49,999.
#### 想法和思考
只需要循环一次记录出现的次数和长度就可以了
```
class Solution {
public:
    int findShortestSubArray(vector<int>& nums) {
        std::unordered_map<int, vector<int>> cache;
        int most_frequency = 0;
        int degree_length = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (cache.count(nums[i]) == 0) {
                cache[nums[i]] = {1, i, i};
                if (most_frequency == 0) {
                    degree_length = 1;
                    most_frequency = 1;
                }
            } else {
                cache[nums[i]][0]++;
                cache[nums[i]][2] = i;
                int length = cache[nums[i]][2] - cache[nums[i]][1] + 1;
                if (cache[nums[i]][0] == most_frequency && length < degree_length) {
                    degree_length = length;
                } 
                else if (cache[nums[i]][0] > most_frequency) {
                    most_frequency = cache[nums[i]][0];
                    degree_length = length;
                }
            }
        }
        return degree_length;
    }
};
```