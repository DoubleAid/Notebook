# Linux进程间通信

## 概括

+ 管道：简单，适用于父子进程间的通信。
+ 消息队列：适用于需要发送结构化消息的场景。
+ 信号：适用于简单的通知和控制。
+ 共享内存：适用于需要高效数据共享的场景。
+ 套接字：适用于需要跨主机通信的场景。
+ 信号量：适用于进程间的同步。
+ 文件和记录锁：适用于防止多个进程同时写入同一个文件。
+ 内存映射文件：适用于需要高效文件访问和进程间通信的场景。

## 信号量

发送端（Sender）

```c
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    // 创建/打开一个命名信号量，初始值为0
    sem_t *sem = sem_open("/my_sem", O_CREAT, 0666, 0);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return 1;
    }

    printf("Sender: 发送信号...\n");
    sem_post(sem);  // 信号量值+1（通知接收端）

    // 关闭信号量（不删除，接收端可能还需使用）
    sem_close(sem);
    return 0;
}
```

接收端（Receiver）

```c
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    // 打开已存在的信号量
    sem_t *sem = sem_open("/my_sem", O_CREAT);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return 1;
    }

    printf("Receiver: 等待信号...\n");
    sem_wait(sem);  // 等待信号量值>0，然后减1

    printf("Receiver: 收到信号！\n");

    // 关闭并删除信号量
    sem_close(sem);
    sem_unlink("/my_sem");
    return 0;
}
```

编译命令：

```bash
gcc sem_sender.c -o sem_sender -lrt
gcc sem_receiver.c -o sem_receiver -lrt
```

## 消息队列

发送端（Sender）

```c
#include <fcntl.h>
#include <mqueue.h>
#include <stdio.h>
#include <string.h>

int main() {
    // 创建或打开消息队列（最大消息数10，消息大小1024字节）
    struct mq_attr attr = {
        .mq_maxmsg = 10,
        .mq_msgsize = 1024
    };
    mqd_t mq = mq_open("/my_queue", O_CREAT | O_WRONLY, 0666, &attr);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        return 1;
    }

    const char *msg = "Hello from Sender!";
    if (mq_send(mq, msg, strlen(msg), 0) == -1) {  // 发送消息（优先级0）
        perror("mq_send");
    } else {
        printf("Sender: 消息已发送\n");
    }

    mq_close(mq);
    return 0;
}
```

接收端（Receiver）

```c
#include <fcntl.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 打开消息队列（只读模式）
    mqd_t mq = mq_open("/my_queue", O_RDONLY);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        return 1;
    }

    // 获取消息属性
    struct mq_attr attr;
    mq_getattr(mq, &attr);
    char *buf = malloc(attr.mq_msgsize);

    // 接收消息（阻塞模式）
    if (mq_receive(mq, buf, attr.mq_msgsize, NULL) == -1) {
        perror("mq_receive");
    } else {
        printf("Receiver: 收到消息: %s\n", buf);
    }

    free(buf);
    mq_close(mq);
    mq_unlink("/my_queue");  // 删除队列
    return 0;
}
```

编译命令：

```bash
gcc mq_sender.c -o mq_sender -lrt
gcc mq_receiver.c -o mq_receiver -lrt
```

## 共享内存

发送端（Sender）

```c
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int main() {
    // 创建共享内存对象并设置大小
    int fd = shm_open("/my_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(fd, 1024);  // 设置共享内存大小为1024字节

    // 映射到进程地址空间
    char *ptr = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // 写入数据
    strcpy(ptr, "Hello from Shared Memory!");
    printf("Sender: 数据已写入共享内存\n");

    // 解除映射（不删除，接收端需要读取）
    munmap(ptr, 1024);
    close(fd);
    return 0;
}
```

接收端（Receiver）

```c
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    // 打开共享内存对象
    int fd = shm_open("/my_shm", O_RDONLY, 0666);

    // 映射到进程地址空间
    char *ptr = mmap(NULL, 1024, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    printf("Receiver: 读取数据: %s\n", ptr);

    // 解除映射并删除共享内存
    munmap(ptr, 1024);
    close(fd);
    shm_unlink("/my_shm");
    return 0;
}
```

编译命令：

```bash
gcc shm_sender.c -o shm_sender -lrt
gcc shm_receiver.c -o shm_receiver -lrt
```

其中 -lrt 是一个链接器选项，用于将实时库（librt）链接到你的程序中。如果你的程序使用了实时库中的函数，如 clock_gettime 或 signalfd，就需要在编译时加上 -lrt 选项。
