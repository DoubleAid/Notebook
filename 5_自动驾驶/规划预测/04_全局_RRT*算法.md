# RRT* 算法

在上一章已经说明了RRT的缺点，适合在开发空间下的路径规划，但因为随机性，路径呈现曲折性，不是最优路径，而RRT*在RRT的基础上做了一些改进，使路径更加平滑，更加合理

RRT*（RRT-star）的核心改进在于通过重布线（Rewiring）和父节点重选机制，使生成的路径逐渐收敛到渐近最优（asymptotically optimal）。

## RRT* 的核心改进

### (1) 父节点重选（Choose Parent）

传统RRT：新节点  $q_{\text{new}}$  直接连接到最近的邻居  $q_{\text{near}}$ ，不考虑路径代价。  

RRT*：在  $q_{\text{new}}$  的邻域半径  r  内，寻找代价最小的父节点（即使不是最近的节点）。  

代价函数：通常为路径长度（如欧氏距离）。  

公式：  

$$
q_{\text{parent}} = \arg\min_{q \in \text{Neighbors}} \left( \text{Cost}(q) + \text{Distance}(q, q_{\text{new}}) \right)
$$

### (2) 重布线（Rewiring）

操作：将  $q_{\text{new}}$  的邻居节点重新连接到  $q_{\text{new}}$ ，如果新路径的代价更小。  

示例：若原路径  $q_{\text{neighbor}} \to q_{\text{parent}}$  的代价为 10，而通过  $q_{\text{new}}$  的新路径代价为 7，则重布线为  $q_{\text{neighbor}} \to q_{\text{new}}$ 。  

效果：动态优化树结构，逐步降低全局路径代价。

## RRT* vs RRT 的关键区别

特性               RRT RRT*

路径质量 可行但不最优 渐近最优（时间足够时收敛到最优）
节点连接策略 连接最近邻节点 邻域内选择代价最小的父节点
优化机制 无 重布线（Rewiring）
计算复杂度 \( O(n) \) \( O(n \log n) \)（邻域查询代价）
适用场景 快速探索可行路径 需要高质量路径的场合

RRT* 的伪代码（关键步骤）

def RRT_star_planning():
    tree = initialize_tree(start)
    while not reach_goal:
        q_rand = random_sample()
        q_near = nearest_neighbor(tree, q_rand)
        q_new = steer(q_near, q_rand, step_size)
        
        if collision_free(q_near, q_new):
            # 1. 父节点重选：在邻域内找代价最小的父节点
            neighbors = find_neighbors(tree, q_new, radius)
            q_parent = q_near
            min_cost = cost(q_near) + distance(q_near, q_new)
            for q in neighbors:
                if cost(q) + distance(q, q_new) < min_cost and collision_free(q, q_new):
                    q_parent = q
                    min_cost = cost(q) + distance(q, q_new)
            tree.add_edge(q_parent, q_new)
            
            # 2. 重布线：优化邻域内其他节点的父节点
            for q in neighbors:
                if cost(q_new) + distance(q_new, q) < cost(q) and collision_free(q_new, q):
                    tree.rewire(q, q_new)  # 将q的父节点改为q_new
    return best_path(tree)

RRT* 的优缺点

优点：
渐近最优性：随着迭代次数增加，路径代价趋近于全局最优。  

适应性：适合高维空间（如机械臂、无人机）和动态环境（结合增量式更新）。  

缺点：
计算代价高：邻域查询和重布线操作增加了实时性负担。  

参数敏感：邻域半径  r  的选择影响性能（太大导致计算慢，太小失去优化意义）。  

改进算法（解决RRT*的缺陷）

Informed RRT*：  

在找到初始路径后，将采样限制在一个椭圆区域内（起点、终点为焦点，初始路径长度为长轴），加速收敛。  
Batch Informed Trees (BIT)：  

结合启发式搜索和批量采样，减少无效探索。  
Dynamic RRT*：  

针对动态障碍物，局部修剪和重优化树结构。

实际应用示例

自动驾驶中的RRT*：
全局路径规划：  

在 Frenet 坐标系下运行 RRT*，横向采样限制在车道内，纵向沿参考路径延伸。  
局部避障：  

当检测到障碍物时，以当前车辆位置为起点，局部运行 RRT* 生成避障轨迹。  

机械臂抓取：
在关节空间（C-space）中，RRT* 生成满足关节角度限制的平滑运动轨迹。

总结
RRT：快速找到可行路径，适合实时性要求高但对路径质量不敏感的场景。  

RRT*：通过重布线和父节点优化，牺牲部分实时性换取最优性，适合精密操作（如手术机器人、自动驾驶）。  

如果需要具体实现（如邻域半径的自适应调整或与Frenet帧的结合），可以进一步探讨！ 🤖✨