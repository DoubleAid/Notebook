# RRT (Rapidly-exploring Randon Tree, 快速探索随机树)

​​RRT​​ 是一种基于采样的路径规划算法，主要用于解决高维空间（如机器人运动规划，无人机避障）或复杂约束环境下的路径搜索问题。与AStar 和 Dijkstra 等基于图搜索的算法不同，RRT 通过随机采样逐步构建一个树状结构，适合处理连续空间、动态障碍物或非完整约束（如机器人运动学限制）

## 核心特点

+ 随机性​​：通过随机采样探索空间，避免遍历所有可能的配置。
+ ​偏向性​​：在随机采样的基础上，向目标点方向偏置（如 5%~10% 的概率直接采样目标点），加速收敛。
+ ​增量式生长​​：每次迭代扩展树的一个节点，逐步覆盖自由空间。

​主要的用途：​机器人路径规划​​（如无人机、机械臂）。高维空间规划​​（如 6 自由度机械臂）。动态避障​​（结合实时环境感知）。

## 算法步骤

```python
def RRT(start, goal, obstacles, max_iter):
    tree = initialize_tree(start)  # 初始化树，根节点为起点
    for _ in range(max_iter):
        q_rand = random_sample()   # 随机采样一个点
        q_near = nearest_neighbor(tree, q_rand)  # 找到树上最近的节点
        q_new = extend(q_near, q_rand, step_size)  # 从 q_near 向 q_rand 扩展一步
        if collision_free(q_near, q_new, obstacles):
            tree.add_node(q_new)   # 添加新节点到树
            if reach_goal(q_new, goal):
                return extract_path(tree, q_new)  # 找到路径
    return None  # 未找到路径
```

### 步骤详解​​
​
+ ​初始化树​​：以起点为根节点。
+ ​随机采样​​：在自由空间中随机选择一个点 q_rand。
+ ​最近邻查询​​：在树上找到离 q_rand 最近的节点 q_near。
+ ​扩展树​​：从 q_near 向 q_rand 方向移动一个固定步长 step_size，得到新节点 q_new。
+ ​碰撞检测​​：检查 q_near 到 q_new 的路径是否与障碍物碰撞。
+ ​添加节点​​：若无碰撞，将 q_new 加入树，并记录父节点为 q_near。
+ ​终止条件​​：
  + 若 q_new 接近目标点，返回路径。
  + 若达到最大迭代次数仍未找到路径，返回失败。

## 局限性

可以看出，RRT在开放空间内的路径规划上是非常方便的，它不需要了解整体环境，只需要在检测碰撞的路径检查是否与障碍物碰撞，但它也有很多缺点：

+ 首先，因为前进方向的随机性，路径可能呈现波动或者曲折性，不是最佳路径
+ RRT在开放空间优异，但是在结构化环境，比如说城市道路路径规划，因为随机采样多是无效点，路径不符合运动约束等，无法使用

在实际使用中通常是组合使用：

+ Hybrid A* + RRT（百度Apollo）​​
  + ​​全局规划​​：Hybrid A* 生成粗略路径（考虑车辆动力学）。
  + ​局部规划​​：RRT在Frenet帧中微调路径，避开动态障碍物。
+ Lattice Planner + RRT（Waymo）​
  + ​生成候选路径​​：Lattice采样一组符合道路曲率的轨迹。
  + ​​RRT优化选择​​：用RRT*在候选集中找到最优无碰撞路径。
