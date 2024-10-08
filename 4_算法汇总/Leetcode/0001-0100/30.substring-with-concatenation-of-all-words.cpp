/*
 * @lc app=leetcode id=30 lang=cpp
 *
 * [30] Substring with Concatenation of All Words
 */

#include <vector>
#include <string>

using namespace std;

// @lc code=start
class Solution {
public:
    // Memory Limit Exceeded

    // vector<int> findSubstring(string s, vector<string>& words) {
    //     // 计划使用递归来做，会简单一点
    //     int count = 0;
    //     vector<int> result;
    //     for (string word : words) count += word.size();
    //     if (count > s.size()) return vector<int>();
    //     for (int i = 0; i < s.size()-count+1; i++) {
    //         if (findBeginSubstring(s.substr(i), words)) {
    //             result.emplace_back(i);
    //         }
    //     }
    //     return result;
    // }

    // bool findBeginSubstring(string s, vector<string>& words) {
    //     if (words.empty()) return true;
    //     for (int i = 0; i < words.size(); i++) {
    //         if (s.substr(0, words[i].size()) == words[i]) {
    //             string remain = s.substr(words[i].size());
    //             vector<string> remain_words = words;
    //             remain_words.erase(remain_words.begin()+i);
    //             if (findBeginSubstring(remain, remain_words)) {
    //                 return true;
    //             }
    //         }
    //     }
    //     return false;
    // }

    vector<int> findSubstring(string s, vector<string>& words) {
    if (words.empty()) return {};

    unordered_map<string, int> word_count;
    for (auto& word : words) word_count[word]++;

    int word_num = words.size();
    int word_len = words[0].size();
    int all_words_len = word_num * word_len;
    vector<int> indices;

    if (all_words_len > s.size()) return indices;

    // 对于每一个可能的起点
    for (int i = 0; i < word_len; i++) {
        int left = i;
        int right = i;
        unordered_map<string, int> has_found;
        int count = 0;

        while (right + word_len <= s.size()) {
            string word = s.substr(right, word_len);
            right += word_len;

            // 如果不是需要找的词，重置窗口
            if (!word_count.count(word)) {
                has_found.clear();
                count = 0;
                left = right;
                continue;
            }

            has_found[word]++;
            count++;

            // 如果当前词的数量超过了需要的数量，调整左边界
            while (has_found[word] > word_count[word]) {
                string tmp = s.substr(left, word_len);
                has_found[tmp]--;
                count--;
                left += word_len;
            }

            // 如果窗口中的词数量等于总词数，记录起点
            if (count == word_num) {
                indices.push_back(left);
                string tmp = s.substr(left, word_len);
                has_found[tmp]--;
                count--;
                left += word_len;
            }
        }
    }

    return indices;
}    
};
// @lc code=end

