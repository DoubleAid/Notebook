# 0x31 RoutineControl

RoutineControl 0x31 用于对主机厂定义的一些特定程序的控制操作 （启动程序，停止程序，请求运行结果）

## 请求格式

| Parameter Name | Information |  Byte Length | Values |
| ---- | ---- | ---- | ---- |
| SID | 请求服务ID | 1 | 31 |
| Subfunction | RoutineControlType | 1 | + 01: 启动程序  02：停止程序 03：请求运行结果 |
| Routine Identifier | 运行的例程ID | 2 | 指定的程序ID 标识符，一般由车厂规定释放 |
| Routine Control Option Record | 例程运行参数 | 不确定 | 程序控制可选参数，用于携带其他信息， 如启动条件，退出条件，一般不需要 |

1. 第2个字节 routine control type 用于指定对程序的操作动作，其可选项如下 （一般用的比较多的是 01 子服务）
    + **01**: start routine 启动程序
    + **02**: stop routine 停止程序
    + **03**: request routine result 请求程序的运行结果

2. 第 3，4 字节 routine identifier 表示指定的程序 ID；这些特定的程序信息一般由车厂规定释放； 一般会在诊断调查表中体现。 比如针对 ECU 的升级， 一般需要指定以下几个特定的程序功能 （具体 ID 由车厂规定）

    程序名 + 功能描述
    + **CheckProgrammingPreconditions(编程条件检查)**: 该程序用于对 ECU 升级条件 （如车速信息等）的判断，在 ECU 的 BootLoader 与 APP程序 中一般要有该程序功能的定义
    + **CheckAppSwAppDataValidity(检查数据的有效性， 如CRC校验)**: 一般用于对升级数据中的每个块信息的CRC校验，ECU 会将接收到的数据 进行 CRC 计算， 并与诊断仪 计算发送过来的CRC进行比对， 确保数据的准确性
    + **CheckProgrammingDependency(编程兼容性的检查)** 用于升级完成后对升级信息的兼容性检查， 例如对升级文件产品型号的检查， 必要的逻辑块是否存在的检查等操作， 一般用以完成所有数据的升级之后

3. 之后的若干数据代表可选记录参数 routine control option record， 一般用于携带其他信息， 如程序的启动条件， 停止条件等， 可以根据实际需要进行使用， 一般很少用到

## 肯定响应格式

| Parameter Name | Information | Byte Length | Values |
| ---- | ---- | ---- | ---- |
| SID | RoutineControlResponseServiceId | 1 | 肯定响应服务ID 71 |
| Subfunction | RoutineControlType | 1 | 与请求服务的SF保持一致 |
| Routine Identifier | 运行的例程 id | 2 | 与请求服务的例程id保持一致 |
| Routine Status Record | 可选参数， 用于反馈ECU的相关信息 | 不确定 |  |

routineStatusRecord 是可选参数， 用于在执行相应程序动作后，返回ECU相关的信息 （如请求 31 02 服务请求停止时 ECU 可以通过该参数返回程序运行的时间等信息），可根据实际需要进行使用，一般很少使用到
