/*
 * @lc app=leetcode id=123 lang=cpp
 *
 * [123] Best Time to Buy and Sell Stock III
 *
 * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/
 *
 * algorithms
 * Hard (47.90%)
 * Likes:    9836
 * Dislikes: 199
 * Total Accepted:    685.5K
 * Total Submissions: 1.4M
 * Testcase Example:  '[3,3,5,0,0,3,1,4]'
 *
 * You are given an array prices where prices[i] is the price of a given stock
 * on the i^th day.
 * 
 * Find the maximum profit you can achieve. You may complete at most two
 * transactions.
 * 
 * Note: You may not engage in multiple transactions simultaneously (i.e., you
 * must sell the stock before you buy again).
 * 
 * 
 * Example 1:
 * 
 * Input: prices = [3,3,5,0,0,3,1,4]
 * Output: 6
 * Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit
 * = 3-0 = 3.
 * Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 =
 * 3.
 * 
 * Example 2:
 * 
 * Input: prices = [1,2,3,4,5]
 * Output: 4
 * Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit
 * = 5-1 = 4.
 * Note that you cannot buy on day 1, buy on day 2 and sell them later, as you
 * are engaging multiple transactions at the same time. You must sell before
 * buying again.
 * 
 * 
 * Example 3:
 * 
 * Input: prices = [7,6,4,3,1]
 * Output: 0
 * Explanation: In this case, no transaction is done, i.e. max profit = 0.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= prices.length <= 10^5
 * 0 <= prices[i] <= 10^5
 * 
 * 
 */
#include <iostream>
#include <vector>

using namespace std;

// @lc code=start
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int buy1 = -prices[0]-1, buy2 = -prices[0]-1;
        int sell1 = 0, sell2 = 0;

        for (int price : prices) {
            buy1 = max(buy1, -price);
            sell1 = max(sell1, buy1 + price);
            buy2 = max(buy2, sell1 - price);
            sell2 = max(sell2, buy2 + price);
        }
        return sell2;
    }
};
// @lc code=end

// 这是一个典型的股票问题，可以用动态规划来解决。既然最多可以进行两次交易，我们可以从一次交易的动态规划拓展到两次交易。让我们一步步引导这个问题的解法。

// 思路
// 可以将问题拆分成几个状态，分别表示在特定时刻持有或不持有股票的最大收益：

// 定义状态：

// 用 buy1 表示完成第一次买入后的最大收益（负数，因为是花钱买股票）。
// 用 sell1 表示完成第一次卖出后的最大收益。
// 用 buy2 表示完成第二次买入后的最大收益。
// 用 sell2 表示完成第二次卖出后的最大收益。
// 状态转移方程：

// 在遍历价格数组的过程中，不断更新这四个状态：
// buy1 = max(buy1, -prices[i])：要么保持不变，要么是当前价格买入。
// sell1 = max(sell1, buy1 + prices[i])：要么保持不变，要么是完成第一次交易。
// buy2 = max(buy2, sell1 - prices[i])：要么保持不变，要么是在第一次交易基础上再次买入。
// sell2 = max(sell2, buy2 + prices[i])：要么保持不变，要么是完成第二次交易。
// 初始状态：

// 将 buy1 和 buy2 初始化为负无穷小（表示还没有买入），而 sell1 和 sell2 初始化为 0（表示还没有收益）。
// 返回结果：

// sell2 最终表示完成最多两次交易的最大收益。
// 实现代码
// 以下是基于以上思路的代码实现：

// python
// 复制代码
// def maxProfit(prices):
//     buy1, sell1 = float('-inf'), 0
//     buy2, sell2 = float('-inf'), 0

//     for price in prices:
//         buy1 = max(buy1, -price)          # 第一次买入的最大收益
//         sell1 = max(sell1, buy1 + price)  # 第一次卖出的最大收益
//         buy2 = max(buy2, sell1 - price)   # 第二次买入的最大收益
//         sell2 = max(sell2, buy2 + price)  # 第二次卖出的最大收益

//     return sell2
// 复杂度分析
// 时间复杂度：O(n)，其中 n 是 prices 的长度，因为我们只需要遍历一次数组。
// 空间复杂度：O(1)，只用了固定的几个变量。
// 通过这种方法，我们可以在线性时间内找到最多两次交易的最大收益。这个问题的关键在于正确地跟踪买入和卖出状态，并确保每次交易之后的状态更新都是最佳的。






// 您说：
// class Solution {
// public:
//     int maxProfit(vector<int>& prices) {
//         vector<vector<int>> dp = vector<vector<int>>(prices.size(), vector<int>(prices.size(), 0));
//         // dp[i][j] 表示 第 i 天到第 j 天的最大收益
//         for (int i = 0; i < prices.size(); i++) {
//             int min_val = prices[i];
//             for (int j = i + 1; j < prices.size(); j++) {
//                 dp[i][j] = max(prices[j] - min_val, dp[i][j-1]);
//                 if (dp[i][j] == 0) dp[i][j] = 0;
//                 min_val = min(min_val, prices[j]);
//             }
//         }
//         int result = 0;
//         for (int i = 0; i < prices.size(); i++) {
//             int left = dp[0][i];

//             int right = i != prices.size() - 1 ? dp[i+1][prices.size()-1] : 0;
//             result = max(left + right, result);
//         }
//         return result;
//     }
// };

// 我这样写应该并不是动态规划，但我不知道怎么写这个动态规划
// ChatGPT 说：
// ChatGPT
// 你目前的代码并没有充分利用动态规划的优势。你构造了一个二维 dp 数组来存储从第 i 天到第 j 天的最大收益，但这样做会导致较高的时间和空间复杂度。对于本题的动态规划解法，我们可以使用更高效的方式，通过四个变量来跟踪每种状态下的最大收益。

// 正确的动态规划思路
// 为了最大化收益，我们需要跟踪四个状态：

// 第一次买入后的最大收益。
// 第一次卖出后的最大收益。
// 第二次买入后的最大收益。
// 第二次卖出后的最大收益。
// 我们可以用一维的动态规划和几个变量来实现，而不需要使用二维数组。这将大大降低时间和空间复杂度。

// 动态规划实现
// 以下是基于上述思路的C++实现：

// cpp
// 复制代码
// class Solution {
// public:
//     int maxProfit(vector<int>& prices) {
//         // 定义四个变量表示四个状态
//         int buy1 = INT_MIN, sell1 = 0;
//         int buy2 = INT_MIN, sell2 = 0;

//         for (int price : prices) {
//             buy1 = max(buy1, -price);            // 第一次买入后的最大收益
//             sell1 = max(sell1, buy1 + price);    // 第一次卖出后的最大收益
//             buy2 = max(buy2, sell1 - price);     // 第二次买入后的最大收益
//             sell2 = max(sell2, buy2 + price);    // 第二次卖出后的最大收益
//         }

//         return sell2;  // 最终返回完成两次交易的最大收益
//     }
// };
// 解释
// buy1: 用于跟踪第一次买入后的最大收益。buy1 初始为负无穷小，在遍历过程中不断更新为更大的值。
// sell1: 用于跟踪第一次卖出后的最大收益。它依赖于 buy1 的值更新。
// buy2: 用于跟踪第二次买入后的最大收益。它依赖于 sell1。
// sell2: 最终我们在 sell2 中记录两次交易完成后的最大收益。
// 时间和空间复杂度
// 时间复杂度：O(n)，因为我们只需要遍历一次 prices 数组。
// 空间复杂度：O(1)，我们只用了固定的四个变量。
// 这个方法充分利用了动态规划，避免了重复计算，从而大大优化了代码的效率。






// 您说：
// 这样不会出现卖比买早的情况吗
// ChatGPT 说：
// ChatGPT
// 不会的。这种动态规划的思路实际上是巧妙地确保了交易的顺序。让我们分析一下为什么这个方法不会出现“卖比买早”的情况：

// 状态转移逻辑
// 在这个方案中，每个状态都是依赖于上一个状态逐步更新的，从而隐式地保证了交易的顺序。

// 第一次买入 (buy1)：代表在到达当前天之前进行过一次买入的最大可能收益。由于 buy1 是直接从 -price 计算的，它不会依赖于后续的卖出行为。

// 第一次卖出 (sell1)：在进行第一次卖出时，sell1 是依赖于 buy1 的，即 sell1 = max(sell1, buy1 + price)。由于 buy1 只会在买入时更新，因此 sell1 也只可能在 buy1 之后更新，这样就保证了第一次卖出一定发生在第一次买入之后。

// 第二次买入 (buy2)：同样，buy2 是基于 sell1 更新的，即 buy2 = max(buy2, sell1 - price)，它代表在进行第一次卖出之后，进行第二次买入的最大收益。因此，第二次买入一定发生在第一次卖出之后。

// 第二次卖出 (sell2)：第二次卖出是基于 buy2 更新的，即 sell2 = max(sell2, buy2 + price)。sell2 的更新只能在 buy2 之后发生，保证了第二次卖出在第二次买入之后。

// 总结
// 通过这种逐步更新的逻辑，每一步的状态更新顺序自然地保证了交易顺序，即：

// 第一次买入 -> 第一次卖出 -> 第二次买入 -> 第二次卖出
// 所以，不会出现卖出早于买入的情况。这也是动态规划的一种“状态依赖”设计，确保了逻辑上交易顺序的正确性。