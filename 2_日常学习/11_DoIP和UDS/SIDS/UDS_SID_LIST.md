# UDS SID 列表

参考链接 https://www.sohu.com/a/239758071_467757

## 分布介绍

| SID Range | Positive Response SID | Info | Protocol Source |
| ---- | ---- | ---- | ---- |
| 00 -- 0F | 40 -- 4F | OBD 服务请求 | ISO15031-5 |
| 10 -- 3E | 50 -- 7E | ISO14229-1 服务请求 | ISO14229-1 |
| 3F | | 保留 | ISO14229-1 保留 |
| 7F | | 否定响应标识 | ISO14229-1 |
| 80 | C0 | 保留 | ISO14229-1 保留 |
| 81 -- 82 | C1 -- C2 | 保留 | ISO14230 保留 |
| 83 -- 87 | C3 -- C7 | ISO14229-1 服务请求 | ISO14229-1 |
| 88 -- 9F | C8 -- DF | 未来扩展(服务请求) | ISO14229-1 保留 |
| A0 -- B9 | E0 -- E0 | 服务请求 | 汽车制造商定义 |
| BA -- BE | FA -- FE | 服务请求 | 系统供应商定义 |
| BF | FF | 保留 | ISO14229-1 保留 |

## ISO14229-1 服务请求

| SID | INFO | Comment | Document |
| ---- | ---- | ---- | ---- |
| 0x10 | Diagnostic Session Control 诊断会话控制 | UDS会使用不同的会话（session），可以用诊断会话控制（Diagnostic Session Control）来切换。可用的服务会依照目前有效的会话而不同。在一开始，控制单元默认是在“默认会话”（Default Session），有定义其他的会话，需要实现的会话会依照设备的种类而不同。 <li>“程序会话”（Programming Session）可以用来上传固件到设备，并更新设备的固件。</li> <li>“扩展诊断会话”（Extended Diagnostic Session）可解锁特定的诊断功能，例如调整传感器等。</li><li>“安全系统诊断会话”（Safety system diagnostic session）用来测试安全相关的诊断机能，例如安全气囊的测试。</li> 此外，也有一些保留的会话识别符，为了汽车生产者及供应商的特殊需求而设计。 | [UDS_0x10_DiagnosticSessionControl](./UDS_0x10_DiagnosticSessionControl.md) |
| 0x11 | ECU Reset ECU 重启 | ECU重置的服务是要重启ECU。依照控制单元硬件以及实现方式的不同，有以下几种不同的重置：<li> “硬重置”模拟电源关闭的重置。</li><li>“关闭锁匙重置”模拟用锁匙将汽车熄火，再开启汽车的点火开关。</li><li>“软重置”初始化特定程序单元以及存储结构。</li>也有一些汽车生产者及供应商定义的特殊数值。 | |
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
| 0x22 | Read Data By Identifier 根据标识符读取资料 | 透过此服务可以读取控制单元中一个或多个的资料。这些资料的种类不限，也可以有不同的长度，例如料号或是软件版本等。也可以读取像是传感器状态之类会变动的值。每一个值会对一个资料标识符（Data Identifier、简称DID），数值从0到65535。会用正常的CAN信号来发送特定ECU使用的资料。DID资料只用在资料请求上，也可以用一些没有ECU使用的DID来发送信息，虽ECU不会使用，但服务工具或软件测试程序可以使用这类的信息。 |
| 0x23 | Read Memory By Address 根据地址读取存储器 | 依给定地址读取物理内存中的值。测试工具可以用此机能来读取软件内部的行为。 |
| 0x24 | Read Scaling Data By Identifier 根据标识符读取ECU标定信息 |  |
| 0x25 |||
| 0x26 |||
| 0x27 | Security Access 安全性访问 | 可以用安全性检查（Security check）来启动大部分的安全关键性服务（security-critical services）。此情形下控制单元会发送“密码种子（seed）”到客户端（电脑或是诊断工具）。客户端再用密码种子计算密钥（key）送回控制单元，以此来解安全关键性服务 |
| 0x28 | Communication Control 通信控制 | 此服务可以关闭控制单元发送以及接收消息的功能。 |
| 0x29 | 认证 Communication Control | 标准在2020年的更新版本，提供一种标准化的方式，可以提供一些安全性访问（0x27）服务无法支持的现代认证方式，包括以PKI为基础的认证交换，以及双向的验证机制。 |
| 0x2A | Read Data By Periodic Identifier | Client 请求周期性读取ECU信息 |
| 0x2B |||
| 0x2C | Dynamically Define Data Identifier ||
| 0x2D | ||
| 0x2E | Write Data By Identifier ||
| 0x2F | Input Output Control By Identifier 输入输出控制 | 此服务可以让外部系统接口透过诊断接口控制输入／输出信号透过设置选择字节，可以设置有关请求的特殊条件，可以设置以下的值：<li>ReturnControlToECU：设备需将信号的控制权送回</li><li>ResetToDefault：测试者试图重置信号，回到系统的默认值</li><li>Freeze Current State：设备需冻结目前的信号，不允许变化</li><li>ShortTermAdjustment：设备需使用目前提供的信号值</li> |
| 0x30 |  ||
| 0x31 | Routine Control 例程控制服务 | 此控制服务程序可以进行各种的服务，有三种不同的信息种类：<li>配合启始信息，可以开始服务。可以定义此信息来确认要执行各动作，或是提示服务已经完成。</li><li>配合停止信息，运行中的服务可以在任何时间下中断。</li><li>第三个选项是查询服务状态的信息</li>可以特别标示启始及结束的信息参数，因此可以实现每一种项目特定的服务。 | [UDS_0x31_RoutineControl](./UDS_0x31_RoutineControl.md) |
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
| 0x3E | Tester Present 在线通知 握手服务 | 若客户端长时间没有交换通信资料，控制单元会自动离开目前的会话，回到“默认会话”，也可能会进入休眠模式。而此一服务的目的就是让控制单元知道客户端仍存在。 |
|||| |
| 0x83 | Access Timing Parameter 访问时序参数 | 在控制器及从机的通信中，需要观察一定的时间，若时间超过此限制，仍没有提交消息，就会假设连接已有问题。可以读取及修改此时间。 | 
| 0x84 | Secured Data Transmission 安全资料传输 | | |
| 0x85 | Control DTC Setting 控制DTC设置 | |
| 0x86 | Response On Event 事件回复 | |
| 0x87 | Link Control 链接控制 | 服务链接控制是用来设置诊断访问的比特率。多半只在中间网关上实现此一机能。 |
