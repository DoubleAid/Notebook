# 问题解决:CoreDate 添加数据后无法打开 preview

错误如下，在添加coredate数据后无法预览
```
Error Domain=FBProcessExit Code=4 "The process crashed." UserInfo={NSLocalizedFailureReason=The process crashed., BSErrorCodeDescription=crash, NSUnderlyingError=0x600002f29dd0 {Error Domain=signal Code=4 "SIGILL(4)" UserInfo={NSLocalizedFailureReason=SIGILL(4)}}}

RemoteHumanReadableError: The operation couldn’t be completed. Transaction failed. Process failed to launch. (process launch failed)

BSTransactionError (1): ==error-description: Process failed to launch. ==precipitating-error: Error Domain=FBProcessExit Code=4 "The process crashed." UserInfo={NSLocalizedFailureReason=The process crashed., BSErrorCodeDescription=crash, NSUnderlyingError=0x600002f29dd0 {Error Domain=signal Code=4 "SIGILL(4)" UserInfo={NSLocalizedFailureReason=SIGILL(4)}}} ==NSLocalizedFailureReason: Transaction failed. Process failed to launch. (process launch failed) ==transaction: <FBApplicationProcessLaunchTransaction: 0x60000186d960> ==error-reason: process launch failed
```

解决方法
1. 执行`xcrun simctl --set previews delete all`命令
2. 重启Xcode