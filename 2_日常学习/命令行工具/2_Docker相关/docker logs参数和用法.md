# Docker logs 命令
docker logs 获取 容器日志
+ -f： 跟踪日志输出
+ --since： 显示某个开始实践的所有日志
+ -t：显示时间戳
+ --tail：仅列出最新N条容器日志

### 用法参考
```
docker logs --since="2016-07-01" --tail=10 my_container
docker logs -f my_container
```