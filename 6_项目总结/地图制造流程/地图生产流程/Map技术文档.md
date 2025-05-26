## class RoadMap
---

### 头文件
```plain
#include "map/hdmap/road_map.h"
```

### 命名空间
```plain
using namespace allride;
using namespace map;
using namespace hdmap_v2;
```



### 初始化RoadMap


| # | 接口名 |
| --- | --- |
| 1 | <font style="color:#DCDCAA;">RoadMap</font><font style="color:#D4D4D4;">(</font><font style="color:#569CD6;">const</font><font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">config</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">ConfigHDMap_RoadMap</font><font style="color:#569CD6;">&</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">config</font><font style="color:#D4D4D4;">);</font> |


### 接口文档
| # | 更新缓存 |
| --- | --- |
| 1 | void updatePosition(const common::math::Vector3d& position); |


| # | 获取 road elements |
| --- | --- |
| 1 | LaneConstPtr getLane(const Id& lane_id) const; |
| 2 | LaneBoundaryConstPtr getLaneBoundary(const Id& lane_boundary_id) const; |
| 3 | LineConstPtr getLine(const Id& line_id) const; |
| 4 | AreaConstPtr getArea(const Id& area_id) const; |
| 5 | SignalRuleConstPtr getSignalRule(const Id& signal_rule_id) const; |
| 6 | StopRuleConstPtr getStopRule(const Id& stop_rule_id) const; |
| 7 | YieldRuleConstPtr getYieldRule(const Id& yield_rule_id) const; |
| 8 | BumperRuleConstPtr getBumperRule(const Id& bumper_rule_id) const; |
| 9 | SpeedRuleConstPtr getSpeedRule(const Id& speed_rule_id) const; |
| 10 | HeightRuleConstPtr getHeightRule(const Id& height_rule_id) const; |
| 11 | ObstacleRuleConstPtr getObstacleRule(const Id& obstacle_rule_id) const; |
| 12 | CrosswalkRuleConstPtr getCrosswalkRule(const Id& crosswalk_rule_id) const; |
| 13 | JunctionRuleConstPtr getJunctionRule(const Id& junction_rule_id) const; |
| 14 | ClearZoneRuleConstPtr getClearZoneRule(const Id& clear_zone_rule_id) const; |
| 15 | DigitRuleConstPtr getDigitRule(const Id& digit_rule_id) const; |
| 16 | CurbConstPtr getCurb(const Id& curb_id) const; |
| 17 | WaitingAreaConstPtr getWaitingArea(const Id& waiting_area_id) const; |
| 18 | ROIRegionConstPtr getROIRegion(const Id& roi_region_id) const; |


| # | 获取 批量road elements |
| --- | --- |
| 1 | std::vector<LaneConstPtr> getLanes(const std::vector<GeoTileId>& tile_ids) const; |
| 2 | 元素同上表 |


| # | 获取 ordered lane rules |
| --- | --- |
| 1 | std::vector<SignalRuleConstPtr> getLaneSignalRules(const Id& lane_id) const; |
| 2 | std::vector<StopRuleConstPtr> getLaneStopRules(const Id& lane_id) const; |
| 3 | std::vector<YieldRuleConstPtr> getLaneYieldRules(const Id& lane_id) const; |
| 4 | std::vector<BumperRuleConstPtr> getLaneBumperRules(const Id& lane_id) const; |
| 5 | std::vector<SpeedRuleConstPtr> getLaneSpeedlRules(const Id& lane_id) const; |
| 6 | std::vector<HeightRuleConstPtr> getLaneHeightRules(const Id& lane_id) const; |
| 7 | std::vector<ObstacleRuleConstPtr> getLaneObstacleRules(const Id& lane_id) const; |
| 8 | std::vector<CrosswalkRuleConstPtr> getLaneCrosswalkRules(const Id& lane_id) const; |
| 9 | std::vector<JunctionRuleConstPtr> getLaneJunctionRules(const Id& lane_id) const; |
| 10 | std::vector<ClearZoneRuleConstPtr> getLaneClearZoneRules(const Id& lane_id) const; |


| # | 获取最近元素 |
| --- | --- |
| 1 | std::vector<std::pair<double, LaneConstPtr>> getNearestLanes(const common::math::Vector3d& center_pos, size_t max_count, double radius = 50.0) const; |
| | std::vector<std::pair<double, LaneConstPtr>> getNearestLanes(const common::math::Vector2d& center_pos,size_t max_count, double radius = 50.0) const; |
| | std::vector<std::pair<double, AreaConstPtr>> getNearestAreas(const common::math::Vector3d& point, size_t max_count, double radius = 50.0) const; |
| | std::vector<std::pair<double, AreaConstPtr>> getNearestAreas(const common::math::Vector2d& point, size_t max_count, double radius = 50.0) const; |
| | std::vector<std::pair<double, AreaConstPtr>> getNearestCrosswalks(const common::math::Vector3d& center_pos3d, size_t max_count) const; |
| | std::vector<std::pair<double, AreaConstPtr>> getNearestCrosswalks(const common::math::Vector2d& center_pos2d, size_t max_count) const; |


| # | 获取关联元素 |
| --- | --- |
| 1 | LaneConstPtr getLeftLane(const Id& lane_id) const; |
| | LaneConstPtr getRightLane(const Id& lane_id) const; |
| | LaneConstPtr getLeftForwardLane(const Id& lane_id) const; |
| | LaneConstPtr getLeftReverseLane(const Id& lane_id) const; |
| | LaneConstPtr getRightForwardLane(const Id& lane_id) const; |
| | LaneConstPtr getRightReverseLane(const Id& lane_id) const; |
| | LaneConstPtr getCrossableLeftLane(const Id& lane_id) const; |
| | LaneConstPtr getCrossableRightLane(const Id& lane_id) const; |
| | std::vector<LaneConstPtr> getSuccessorLanes(const Id& lane_id) const; |
| | std::vector<LaneConstPtr> getPredecessorLanes(const Id& lane_id) const; |
| | std::vector<LaneSequence> getLaneSequencesByDepth(const Id& lane_id, size_t depth, bool forward = true) const; |
| | std::vector<LaneSequence> getLaneSequencesByLength(const Id& lane_id, double length, bool forward = true) const; |
| | std::shared_ptr<Refline> getRefline(const LaneSequence& lane_sequence, double expected_len = 0.0, bool forward = true) const; |
| | std::vector<std::shared_ptr<Refline>> getReflines(const std::vector<LaneSequence>& lane_sequences, double expected_len = 0.0, bool forward = true) const; |
| | std::shared_ptr<Refline> getSmoothedRefline(const LaneSequence& lane_sequence, std::shared_ptr<Refline> prev_refline = nullptr) const; |
| | std::vector<std::shared_ptr<Refline>> getSmoothedReflines(const std::vector<LaneSequence>& lane_sequences,std::shared_ptr<Refline> prev_refline = nullptr) const; |
| | std::shared_ptr<Refline> getSmoothedReflineV2(const LaneSequence& lane_sequence, const Id* source_id = nullptr) const; |
| | std::vector<std::shared_ptr<Refline>> getSmoothedReflinesV2(const std::vector<LaneSequence>& lane_sequences,const Id* source_id = nullptr) const; |
| | std::vector<std::shared_ptr<Refline>> getForkedReflines(const LaneSequence& lane_sequence, double primary_len,double fork_len, bool fork_forward) const; |
| | double getCenterlineLength(const Id& id) const; |


| | 获取tile |
| --- | --- |
| 1 | std::vector<GeoTileId> getAllTileIds() const; |
| | RoadMapTileConstPtr getTile(const GeoTileId& tile_id) const; |
| | RoadMapItems getAllItems() const; |
| | RoadMapItems getItemsInRange(const common::math::Vector2d& min_corner, const common::math::Vector2d& max_corner) const; |


| | 更新缓存 |
| --- | --- |
| 1 | void updatePosition(const common::math::Vector3d& position); |


## class Router
---

### 头文件
```plain
#include "map/router/router.h"
```

### 命名空间
```plain
using namespace allride;
using namespace map;
using namespace router_v2;
```



### 初始化router


| # | 接口名 |
| --- | --- |
| 1 | <font style="color:#DCDCAA;">Router</font><font style="color:#D4D4D4;">(</font><font style="color:#569CD6;">const</font><font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">config</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">ConfigHDMap_Router</font><font style="color:#569CD6;">&</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">config</font><font style="color:#D4D4D4;">)</font> |
| 2 | <font style="color:#DCDCAA;">Router</font><font style="color:#D4D4D4;">(</font><font style="color:#569CD6;">const</font><font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">config</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">ConfigHDMap_Router</font><font style="color:#569CD6;">&</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">config</font><font style="color:#D4D4D4;">,  </font><font style="color:#4EC9B0;">std</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">shared_ptr</font><font style="color:#D4D4D4;"><</font><font style="color:#4EC9B0;">hdmap_v2</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">RoadMap</font><font style="color:#D4D4D4;">> </font><font style="color:#9CDCFE;">road_map</font><font style="color:#D4D4D4;">)</font> |


### 接口文档


| # | 接口名 |
| --- | --- |
| 1 | <font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">map</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">HDMapRoute</font><font style="color:#D4D4D4;"> </font><font style="color:#DCDCAA;">navigate</font><font style="color:#D4D4D4;">(</font><font style="color:#569CD6;">const</font><font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">math</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">Vector3d</font><font style="color:#569CD6;">&</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">from</font><font style="color:#D4D4D4;">, </font><font style="color:#569CD6;">const</font><font style="color:#D4D4D4;"> </font><font style="color:#4EC9B0;">proto</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">math</font><font style="color:#D4D4D4;">::</font><font style="color:#4EC9B0;">Vector3d</font><font style="color:#569CD6;">&</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">to</font><font style="color:#D4D4D4;">, </font><font style="color:#569CD6;">bool</font><font style="color:#D4D4D4;"> </font><font style="color:#9CDCFE;">viz</font><font style="color:#D4D4D4;"> = </font><font style="color:#569CD6;">true</font><font style="color:#D4D4D4;">);</font> |




```plain

```

