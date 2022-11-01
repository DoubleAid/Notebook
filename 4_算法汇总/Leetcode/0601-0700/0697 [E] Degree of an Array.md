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