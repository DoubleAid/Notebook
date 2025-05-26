# AStar 算法

在带权图中找到从​​起点到目标点​​的最短路径，通过启发式函数加速搜索。比Dijkstra更高效（减少搜索范围），但需已知目标位置。
​​
​​核心思想​​：

+ 综合实际代价 g(n) 和启发式估计 h(n)，优先搜索最有希望的节点。
+ 公式：f(n) = g(n) + h(n)，其中：
  + g(n)：从起点到节点 n 的实际代价。
  + h(n)：从节点 n 到目标的​​预估代价​​（需满足​​可采纳性​​，即 h(n) ≤ 真实代价）。

## 输入​​

带权图 G=(V, E)，起点 start，目标点 goal。
启发式函数 h(n)（如欧氏距离、曼哈顿距离）。

## ​​输出​​

从 start 到 goal 的最短路径及总代价。

## 算法步骤

+ ​初始化​​：
  + 设置起点的 g(start) = 0，f(start) = h(start)。
  + 其他节点 g(n) = ∞，f(n) = ∞。
  + 创建优先队列（最小堆）OPEN，按 f(n) 排序，初始放入 start。
  + 记录父节点 prev 和已访问集合 CLOSED。
​
+ ​主循环​​：
  + ​While​​ OPEN 不为空：
    + 从 OPEN 中取出 f(n) 最小的节点 current。
    + If​​ current == goal：
      + 回溯 prev 重建路径，返回结果。
    + 将 current 加入 CLOSED（标记为已访问）。
    + For each​​ 邻居 neighbor of current：
      + If​​ neighbor 在 CLOSED 中：跳过。
      + 计算临时 g_temp = g(current) + cost(current, neighbor)。
      + ​​If​​ g_temp < g(neighbor)（发现更优路径）：
        + 更新 g(neighbor) = g_temp。
        + 更新 f(neighbor) = g(neighbor) + h(neighbor)。
        + 记录 prev[neighbor] = current。
​​      + If​​ neighbor 不在 OPEN 中：将其加入。
​
+ ​终止​​：
  + 若 OPEN 为空且未找到目标，则路径不存在。
