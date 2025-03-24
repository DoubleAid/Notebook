# **Linux进程调度**

## **1. 调度机制**

- **调度器核心逻辑**：
  - **完全公平调度器（CFS）**：默认调度策略，通过红黑树跟踪进程的虚拟运行时间（`vruntime`），确保所有进程公平获取CPU时间。
  - **实时调度策略**：
    - `SCHED_FIFO`：先进先出，不设时间片，高优先级进程独占CPU。
    - `SCHED_RR`：轮转调度，每个进程分配固定时间片。
  - **调度类（Scheduling Classes）**：优先级顺序为`Stop-Sched` > `Deadline` > `RT` > `CFS` > `Idle`。

- **多核负载均衡**：
  - 通过`load_balance()`函数将任务迁移到空闲核心。
  - 使用`sched_domain`层级结构管理CPU拓扑。

## **2. 进程调度算法**

| **算法**       | **特点**                                                                 | **适用场景**               |
|----------------|--------------------------------------------------------------------------|--------------------------|
| **CFS**        | 基于虚拟运行时间，红黑树排序，公平分配CPU                                | 通用场景（默认策略）       |
| **SCHED_FIFO** | 无时间片，高优先级进程独占CPU，直到主动释放                              | 实时任务（如工业控制）     |
| **SCHED_RR**   | 轮转时间片（默认100ms），同优先级进程轮流执行                            | 实时任务需共享CPU          |
| **Deadline**   | 为任务设置绝对截止时间，优先调度最紧迫的任务                             | 严格时限任务（如机器人控制）|

---

## 我有四个核的CPU，我希望有一个核执行A程序，一个核执行B，C，D程序，剩下两个核负载均衡，执行其他进程

不是的，`taskset` 和 `cgroups` 是两种不同的工具，可以独立使用来设置CPU亲和性。你不需要同时使用它们来实现CPU亲和性设置。根据你的需求，可以选择其中一种方法即可。

### 使用 `taskset` 设置CPU亲和性

`taskset` 是一个简单的命令行工具，用于设置或获取进程的CPU亲和性。它可以直接在命令行中使用，非常适合快速设置特定进程的CPU亲和性。

#### 示例：将程序绑定到特定CPU核心

1. **将A程序绑定到CPU 0**：

   ```bash
   taskset -c 0 ./A_program
   ```

2. **将B、C、D程序绑定到CPU 1**：

   ```bash
   taskset -c 1 ./B_program
   taskset -c 1 ./C_program
   taskset -c 1 ./D_program
   ```

### 使用 `cgroups` 设置CPU亲和性

`cgroups`（控制组）是一个更强大的资源管理工具，可以对进程组进行细粒度的资源控制，包括CPU亲和性、内存限制等。`cgroups` 通常用于更复杂的资源管理场景。

#### 示例：使用 `cgroups` 设置CPU亲和性

1. **创建 `cgroups`**：

   ```bash
   sudo cgcreate -g cpuset:/mygroup
   ```

2. **设置CPU亲和性**：
   - 将CPU 0分配给A程序：

     ```bash
     echo 0 > /sys/fs/cgroup/cpuset/mygroup/cpuset.cpus
     ```

   - 将CPU 1分配给B、C、D程序：

     ```bash
     echo 1 > /sys/fs/cgroup/cpuset/mygroup/cpuset.cpus
     ```

3. **将进程添加到 `cgroups`**：
   - 启动A程序并将其PID添加到CPU 0的 `cgroup`：

     ```bash
     ./A_program &
     echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
     ```

   - 启动B、C、D程序并将其PID添加到CPU 1的 `cgroup`：

     ```bash
     ./B_program &
     echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
     ./C_program &
     echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
     ./D_program &
     echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
     ```

### 选择哪种方法

- **简单场景**：如果你只需要快速设置特定进程的CPU亲和性，使用 `taskset` 是最简单的方法。
- **复杂场景**：如果你需要对多个进程进行细粒度的资源管理，或者需要限制内存等其他资源，使用 `cgroups` 更为合适。

### 示例代码

#### 使用 `taskset`

```bash
# 启动A程序并绑定到CPU 0
taskset -c 0 ./A_program &

# 启动B、C、D程序并绑定到CPU 1
taskset -c 1 ./B_program &
taskset -c 1 ./C_program &
taskset -c 1 ./D_program &
```

#### 使用 `cgroups`

```bash
# 创建cgroup
sudo cgcreate -g cpuset:/mygroup

# 设置CPU亲和性
echo 0 > /sys/fs/cgroup/cpuset/mygroup/cpuset.cpus

# 启动A程序并将其PID添加到CPU 0的cgroup
./A_program &
echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks

# 设置CPU亲和性
echo 1 > /sys/fs/cgroup/cpuset/mygroup/cpuset.cpus

# 启动B、C、D程序并将其PID添加到CPU 1的cgroup
./B_program &
echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
./C_program &
echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
./D_program &
echo $! > /sys/fs/cgroup/cpuset/mygroup/tasks
```

### 总结

- **`taskset`**：适用于简单场景，快速设置特定进程的CPU亲和性。
- **`cgroups`**：适用于复杂场景，对多个进程进行细粒度的资源管理。

## 如何设置某个程序的内存空间呢

在Linux系统中，使用`cgroups`（控制组）可以有效地限制进程的内存使用。以下是如何使用`cgroups`来限制特定进程组的内存使用。

### 创建和配置cgroup

1. **创建cgroup**：

   首先，你需要创建一个cgroup来管理内存。例如，创建一个名为`mygroup`的cgroup：

   ```bash
   sudo cgcreate -g memory:/mygroup
   ```

2. **设置内存限制**：
   你可以通过设置`memory.limit_in_bytes`来限制cgroup的内存使用。例如，限制内存使用不超过1GB：

   ```bash
   echo $((1024*1024*1024)) > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
   ```

3. **设置交换空间限制**（可选）：
   如果你也想限制交换空间的使用，可以设置`memory.memsw.limit_in_bytes`。例如，限制交换空间使用不超过2GB：

   ```bash
   echo $((2*1024*1024*1024)) > /sys/fs/cgroup/memory/mygroup/memory.memsw.limit_in_bytes
   ```

### 将进程分配到cgroup

1. **启动进程并分配到cgroup**：
   你可以使用`cgexec`命令来启动进程，并将其分配到特定的cgroup。例如，启动一个进程并将其分配到`mygroup`：

   ```bash
   cgexec -g memory:mygroup ./my_program
   ```

2. **手动将现有进程分配到cgroup**：
   如果你已经有一个正在运行的进程，你可以手动将其分配到cgroup。首先，找到进程的PID，然后将其添加到cgroup：

   ```bash
   echo <PID> > /sys/fs/cgroup/memory/mygroup/tasks
   ```

### 监控内存使用

1. **查看内存使用情况**：
   你可以查看cgroup的内存使用情况，以确保限制生效。例如，查看当前cgroup的内存使用情况：

   ```bash
   cat /sys/fs/cgroup/memory/mygroup/memory.usage_in_bytes
   ```

2. **查看内存限制**：
   查看当前设置的内存限制：

   ```bash
   cat /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
   ```

### 示例：完整流程

以下是一个完整的示例，展示如何创建cgroup，设置内存限制，并启动一个进程：

```bash
# 创建cgroup
sudo cgcreate -g memory:/mygroup

# 设置内存限制为1GB
echo $((1024*1024*1024)) > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes

# 设置交换空间限制为2GB（可选）
echo $((2*1024*1024*1024)) > /sys/fs/cgroup/memory/mygroup/memory.memsw.limit_in_bytes

# 启动一个进程并将其分配到cgroup
cgexec -g memory:mygroup ./my_program
```

通过这些步骤，你可以有效地限制特定进程组的内存使用，确保系统资源的合理分配和使用。

## 实现不同权重的程序的负载均衡

要实现基于权重的负载均衡，可以使用Linux的`cgroups`（控制组）来分配CPU资源。`cgroups`的`cpu`子系统允许你为进程分配CPU时间片，从而实现负载均衡。以下是一个具体的实现步骤：

### 1. 创建和配置`cgroups`

1. **挂载`cgroups`文件系统**：

   ```bash
   sudo mount -t cgroup -o cpu,cpuset cpu /sys/fs/cgroup/cpu
   sudo mount -t cgroup -o cpuset cpuset /sys/fs/cgroup/cpuset
   ```

2. **创建`cgroups`**：
   为每个程序创建一个单独的`cgroup`。例如，创建12个`cgroup`：

   ```bash
   for i in {1..12}; do
       sudo cgcreate -g cpu,cpuset:/mygroup$i
   done
   ```

### 2. 设置CPU亲和性和权重

1. **设置CPU亲和性**：
   假设你有4个CPU核心（CPU 0-3），你可以将这些核心分配给不同的`cgroup`。例如，将CPU 0-3分配给所有`cgroup`：

   ```bash
   for i in {1..12}; do
       echo 0-3 > /sys/fs/cgroup/cpuset/mygroup$i/cpuset.cpus
   done
   ```

2. **设置CPU权重**：
   使用`cpu.shares`来设置每个`cgroup`的CPU权重。权重值越大，分配的CPU时间片越多。例如，设置权重为10, 8, 8, 6, 6, 6, 4, 4, 4, 4, 4, 4：

   ```bash
   echo 1000 > /sys/fs/cgroup/cpu/mygroup1/cpu.shares
   echo 800 > /sys/fs/cgroup/cpu/mygroup2/cpu.shares
   echo 800 > /sys/fs/cgroup/cpu/mygroup3/cpu.shares
   echo 600 > /sys/fs/cgroup/cpu/mygroup4/cpu.shares
   echo 600 > /sys/fs/cgroup/cpu/mygroup5/cpu.shares
   echo 600 > /sys/fs/cgroup/cpu/mygroup6/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup7/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup8/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup9/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup10/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup11/cpu.shares
   echo 400 > /sys/fs/cgroup/cpu/mygroup12/cpu.shares
   ```

### 3. 启动程序并分配到`cgroups`

1. **启动程序并分配到`cgroups`**：
   使用`cgexec`命令启动每个程序，并将其分配到相应的`cgroup`。例如：

   ```bash
   cgexec -g cpu,cpuset:mygroup1 ./program1 &
   cgexec -g cpu,cpuset:mygroup2 ./program2 &
   cgexec -g cpu,cpuset:mygroup3 ./program3 &
   cgexec -g cpu,cpuset:mygroup4 ./program4 &
   cgexec -g cpu,cpuset:mygroup5 ./program5 &
   cgexec -g cpu,cpuset:mygroup6 ./program6 &
   cgexec -g cpu,cpuset:mygroup7 ./program7 &
   cgexec -g cpu,cpuset:mygroup8 ./program8 &
   cgexec -g cpu,cpuset:mygroup9 ./program9 &
   cgexec -g cpu,cpuset:mygroup10 ./program10 &
   cgexec -g cpu,cpuset:mygroup11 ./program11 &
   cgexec -g cpu,cpuset:mygroup12 ./program12 &
   ```

### 4. 验证设置

1. **查看CPU亲和性**：
   使用`taskset`命令查看每个程序的CPU亲和性：

   ```bash
   taskset -p $(pgrep program1)
   taskset -p $(pgrep program2)
   taskset -p $(pgrep program3)
   taskset -p $(pgrep program4)
   taskset -p $(pgrep program5)
   taskset -p $(pgrep program6)
   taskset -p $(pgrep program7)
   taskset -p $(pgrep program8)
   taskset -p $(pgrep program9)
   taskset -p $(pgrep program10)
   taskset -p $(pgrep program11)
   taskset -p $(pgrep program12)
   ```

2. **查看CPU权重**：
   查看每个`cgroup`的CPU权重：

   ```bash
   cat /sys/fs/cgroup/cpu/mygroup1/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup2/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup3/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup4/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup5/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup6/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup7/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup8/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup9/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup10/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup11/cpu.shares
   cat /sys/fs/cgroup/cpu/mygroup12/cpu.shares
   ```

### 总结

通过使用`cgroups`的`cpu`和`cpuset`子系统，你可以为每个程序分配特定的CPU核心和权重，从而实现基于权重的负载均衡。每个程序的CPU时间片将根据其权重比例分配，确保系统资源的合理利用。
