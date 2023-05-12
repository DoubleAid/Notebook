# UDS SID 列表

参考链接 https://www.sohu.com/a/239758071_467757

## 分布介绍

| SID Range | Info | Protocol Source |
| ---- | ---- | ---- |
| 00 -- 0F | OBD 服务请求 | ISO15031-5 |
| 10 -- 3E | ISO14229-1 服务请求 | ISO14229-1 |
| 3F | 保留 | ISO14229-1 保留 |
| 40 -- 4F | OBD 肯定响应 | ISO15031-5 |
| 50 -- 7E | ISO14229-1 肯定响应 | ISO14229-1 |
| 7F | 否定响应标识 | ISO14229-1 |
| 80 | 保留 | ISO14229-1 保留 |
| 81 -- 82 | 保留 | ISO14230 保留 |
| 83 -- 87 | ISO14229-1 服务请求 | ISO14229-1 |
| 88 -- 9F | 未来扩展(服务请求) | ISO14229-1 保留 |
| A0 -- B9 | 服务请求 | 汽车制造商定义 |
| BA -- BE | 服务请求 | 系统供应商定义 |
| BF | 保留 | ISO14229-1 保留 |
| C0 | 保留(肯定响应) | ISO14229-1 保留 |
| C1 -- C2 | 保留(肯定响应) | ISO14230 保留 |
| C3 -- C7 | ISO14229-1 ||
| C8 -- DF | 未来扩展(肯定相应) | ISO14229-1 保留 |
| E0 -- F9 | 请求肯定响应 | 汽车制造商定义 |
| FA -- FE | 请求肯定响应 | 系统供应商定义 |
| FF | 保留(肯定响应) | ISO14229-1 保留 |

## ISO14229-1 服务请求

| SID | INFO | Comment | Document |
| ---- | ---- | ---- | ---- |
| 0x10 | Diagnostic Session Control | 诊断会话控制服务 | [UDS_0x10_DiagnosticSessionControl](./UDS_0x10_DiagnosticSessionControl.md) |
| 0x11 | ECU Reset | client 请求 ECU 重启 | |
| 0x12 |  || |
| 0x13 |  || |
| 0x14 | Clear Diagnostic Information | 清除诊断信息 |
| 0x15 |||
| 0x16 |||
| 0x17 |||
| 0x18 |||
| 0x19 | Read DTC Information ||
| 0x1A |||
| 0x1B |||
| 0x1C |||
| 0x1D |||
| 0x1E |||
| 0x1F |||
| 0x20 |||
| 0x21 |||
| 0x22 | Read Data By Identifier | Client 请求根据数据表示读取ECU数据 |
| 0x23 | Read Memory By Address | Client 请求根据数据地址读取ECU数据 |
| 0x24 | Read Scaling Data By Identifier | Client 请求根据数据表示读取ECU标定信息 |
| 0x25 |||
| 0x26 |||
| 0x27 | Security Access | Client 请求 解锁 ECU 安全保护状态 |
| 0x28 | Communication Control | Client 请求控制 ECU 通信 |
| 0x29 |  ||
| 0x2A | Read Data By Periodic Identifier | Client 请求周期性读取ECU信息 |
| 0x2B |||
| 0x2C | Dynamically Define Data Identifier ||
| 0x2D | ||
| 0x2E | Write Data By Identifier ||
| 0x2F | Input Output Control By Identifier | 输入输出控制 | |
| 0x30 |  ||
| 0x31 | Routine Control | 例程控制服务 | [UDS_0x31_RoutineControl](./UDS_0x31_RoutineControl.md) |
| 0x32 |||
| 0x33 |||
| 0x34 | Request Download | 下载数据 | |
| 0x35 | Request Upload || |
| 0x36 | Transfer Data || |
| 0x37 | Request Transfer Exit || |
| 0x38 | Request File Transfer || |
| 0x39 ||| |
| 0x3A ||| | 
| 0x3B ||| |
| 0x3C |  ||
| 0x3D | Write Memory By Address | 根据地址写入内存 | |
| 0x3E | Tester Present | Client 在线通知 握手服务 | |
|||| |
| 0x83 | Access Timing Parameter | Client 请求控制ECU通信周期 |
| 0x84 | Secured Data Transmission | Client 请求执行安全保护的数据传输 |
| 0x85 | Control DTC Setting | Client请求设置ECU DTC |
| 0x86 | Response On Event | Client 请求 ECU 执行特定事件 |
| 0x87 | Link Control | Client 请求控制ECU通信波特率 |
