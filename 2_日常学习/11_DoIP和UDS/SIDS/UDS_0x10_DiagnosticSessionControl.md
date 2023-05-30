# Diagnostc Session Control

0x10诊断会话控制服务用于实现Client请求切换ECU的诊断模式。ECU上电后处于默认模式（Default diagnostic session），在默认模式中可实现的诊断服务是有一定限制的，
比如无法解锁安全模式，则Bootloader、部分RoutineControl、部分数据写入就无法实现，直到通过0x10诊断会话切换ECU至non-default session。

## 请求格式

| Parameter Name | Information | Byte Length | Values |
| ---- | ---- | ---- | ---- |
| SID |  | 1 Byte | 0x10 |
| Session || 1 Byte | 0x01/0x02/0x03 |

