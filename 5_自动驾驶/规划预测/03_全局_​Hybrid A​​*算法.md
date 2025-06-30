# Hybrid A* 混合AStar算法

Hybrid A* 是一种结合 离散A 和 连续状态空间搜索 的路径规划算法，专为车辆、机器人等 连续运动系统 设计。Hybrid A*通过平衡离散搜索和连续建模，成为复杂运动系统中路径规划的首选算法之一。

## 核心思想

它在传统A*的基础上引入两个关键改进：

+ 连续状态表示：使用连续坐标（如车辆位姿 (x, y, θ)）而非离散网格。
+ 运动学约束：考虑车辆的非完整约束（如转向半径限制），生成物理可行的路径。

## 算法流程

### 步骤 1：状态表示

状态节点：s = (x, y, θ, v)（坐标、朝向、速度等）。
离散化：将连续状态空间按分辨率离散（如 x±Δx, θ±Δθ），但保持连续计算。

### 步骤 2：启发式函数

**启发式设计：**

+ 非完整约束启发式：考虑车辆最小转弯半径（Reeds-Shepp曲线或Dubins路径）。
+ 障碍物忽略启发式：在无障碍时快速收敛，接近经典A*。

### 步骤 3：节点扩展

运动基元（Motion Primitives）：预定义符合车辆运动学的局部路径（如圆弧+直线组合）。

```python
# 示例：车辆运动基元（转向角δ，速度v）
primitives = [(δ1, v1), (δ2, v2), ...]  # 覆盖转向、直行、倒车等
```

### 步骤 4：代价计算

代价函数：f(s) = g(s) + h(s)

+ g(s)：从起点到当前状态的实际代价（如路径长度、转向惩罚）。
+ h(s)：启发式估计到目标的代价（通常用Reeds-Shepp距离）。

### 步骤 5：路径优化

后处理：对原始路径进行平滑（样条插值、数值优化），确保连续性和可行性。

## 与经典A*的对比

| 特性 | 经典A | Hybrid A |
| ---- | ---- | ---- |
| 状态空间 | 完全离散（网格）| 连续+离散混合 |
| 运动约束 | 忽略（八连通/四连通）| 显式建模（如转向半径）|
| 计算效率 | 高（简单网格）| 较低（需处理连续状态）|
| 路径可行性 | 可能不满足动力学 | 物理可行（适合车辆/机器人）|
| 应用场景 | 游戏NPC、网格地图 | 自动驾驶、移动机器人 |

## 应用场景

+ 自动驾驶：停车场泊车、狭窄道路调头。
+ 移动机器人：仓库AGV路径规划。
+ 无人机：考虑动力学约束的航迹规划。

## 代码示例（Python伪代码）

```python
import numpy as np

class HybridAStar:
    def __init__(self, grid_map):
        self.map = grid_map  # 占据栅格地图
        self.primitives = self._generate_primitives()  # 运动基元

    def plan(self, start, goal):
        open_set = {start}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda s: f_score[s])
            if self.reached_goal(current, goal):
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for primitive in self.primitives:
                neighbor = self.apply_primitive(current, primitive)
                if not self.is_valid(neighbor):
                    continue
                tentative_g = g_score[current] + self.cost(primitive)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    open_set.add(neighbor)
        return None
```

## 改进与变种

+ Kinodynamic A*：进一步整合动力学约束（加速度、曲率连续性）。
+ Lattice Planner：基于状态晶格（State Lattice）的Hybrid A*变种。
+ Parallel Hybrid A*：GPU加速运动基元计算。

## 局限性

+ 计算复杂度：连续状态搜索导致节点扩展数剧增。
+ 启发式依赖：Reeds-Shepp/Dubins路径可能低估实际代价。

## 工具推荐&开源实现

+ https://github.com/ros-planning/navigation
+ https://github.com/maxspahn/hybrid_astar
+ 可视化工具：MATLAB的Automated Driving Toolbox。
