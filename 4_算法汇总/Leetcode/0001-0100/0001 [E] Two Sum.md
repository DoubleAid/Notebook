# 001 [E] Two Sum 两数之和

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

#### Example 1

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
```

#### Example 2

```
Input: nums = [3,2,4], target = 6
Output: [1,2]
```

#### Example 3

```
Input: nums = [3,3], target = 6
Output: [0,1]
```

#### Constraints

+ 2 <= nums.length <= 104
+ -109 <= nums[i] <= 109
+ -109 <= target <= 109
+ Only one valid answer exists.

#### 思路
申请target+1长度的空间，全置为 -1
对于 i ；
+ 先查看 target-i 位置是否为 -1， 是则将taiget-i 标注为 自己当前坐标
+ 不是则返回 target-i 标注坐标和当前坐标

```
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        enmuq={}
        enums=enumerate(nums) #枚举化
        for i, each in enums:
            if target-each in enmuq:
                return [enmuq[target-each], i]
            enmuq[each] = i
        return []
```