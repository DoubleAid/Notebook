# KDTree 和 iKDTree 的区别

## KDTree

KDTree 是一种静态的经典二叉树结构，主要用于近邻搜索和范围搜索

数据结构

```cpp
struct KDTreeNode {
    vector<float> point;    // 点的坐标
    int split_dim;          // 分割的维度
    KDTreeNode *left;       // 左子树
    KDTreeNode *right;      // 右子树
};
```

构建过程

1. 递归分割：如果桶内点的个数大于阈值，选择方差最大的维度进行分割
2. 选择中值：通常选用中位数作为分割点
3. 构建二叉树：递归构建左右子树
4. 时间复杂度：O(nlogn)构建，O(logn)查询

但不能动态更新，不适合增量式的slam，不平衡时会性能下降

## iKDTree

iKDTree是一种支持增量更新的动态数据结构，由FAST-LIO论文提出，用于实时激光雷达SLAM的地图维护

数据结构

```cpp
struct iKDTreeNode {
    vector<int> point;
    int split_dim;
    iKDTreeNode *left;
    iKDtreeNode *right;

    // 增量特性
    bool deleted;                // 软删除标记
    int point_count;             // 子树中的点数
    vector<iKDTreeNode*> points; // 叶节点存储多个点
};
```

主要的改进

1. 懒惰删除和重建，删除节点时，将节点标记为删除，并记录删除的点数。在后面通过动态平衡机制，将子树重新构建。

2. 动态平衡机制, 如果子树节点数超过阈值，则进行平衡

3. 批量插入优化： 传统上会煮点逐点插入，每次的复杂度时 O(nlogn)，而ikdtree会都累积到缓冲区，在达到阈值时批量重建

4. 子树局部重建：和构建全局KDTree一样，通过创建多个叶节点 -> 选个分割维度（最大方差）-> 分割点集 -> 递归重建

## 代码实现

KD-Tree 实现

```cpp
class KDTree {
private:
    KDTreeNode* root;
    
    KDTreeNode* build(vector<Point>& points, int depth) {
        if (points.empty()) return nullptr;
        
        int k = points[0].size();
        int axis = depth % k;
        
        // 按当前维度排序
        sort(points.begin(), points.end(), 
             [axis](Point& a, Point& b) { return a[axis] < b[axis]; });
        
        int median = points.size() / 2;
        KDTreeNode* node = new KDTreeNode(points[median]);
        
        vector<Point> left(points.begin(), points.begin() + median);
        vector<Point> right(points.begin() + median + 1, points.end());
        
        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);
        
        return node;
    }
};
```

iKD-Tree 实现（简化）

```cpp
class iKDTree {
private:
    iKDTreeNode* root;
    vector<Point> insert_buffer;
    int rebuild_counter = 0;
    const int REBUILD_THRESHOLD = 100;
    
    // 增量插入
    void incremental_insert(Point point) {
        insert_buffer.push_back(point);
        rebuild_counter++;
        
        if (rebuild_counter >= REBUILD_THRESHOLD) {
            rebuild_affected_subtree(root, insert_buffer);
            insert_buffer.clear();
            rebuild_counter = 0;
        }
    }
    
    // 子树重建
    void rebuild_affected_subtree(iKDTreeNode*& node, vector<Point>& points) {
        if (node == nullptr || node->deleted) {
            node = build_new_subtree(points);
            return;
        }
        
        // 检查是否需要重建
        float imbalance = calculate_imbalance(node);
        if (imbalance > 0.7) {
            // 收集子树所有点
            vector<Point> subtree_points = collect_points(node);
            subtree_points.insert(subtree_points.end(), 
                                  points.begin(), points.end());
            
            // 重建
            node = build_new_subtree(subtree_points);
        } else {
            // 递归处理左右子树
            vector<Point> left_points, right_points;
            for (auto& p : points) {
                if (p[node->split_dim] < node->split_val)
                    left_points.push_back(p);
                else
                    right_points.push_back(p);
            }
            
            rebuild_affected_subtree(node->left, left_points);
            rebuild_affected_subtree(node->right, right_points);
        }
    }
};
```