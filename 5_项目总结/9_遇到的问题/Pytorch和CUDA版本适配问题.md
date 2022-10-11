RTX3050ti、3060等系列显卡正确安装cuda、cudnn,配置Pytorch深度学习环境（避免由于版本不适应导致重装）
如果你遇到以下问题，请认真看完，相信会有收获
```
GeForce RTX 3070 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
If you want to use the GeForce RTX 3070 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
```
所需的下载准备

重点介绍在rtx30系列安装正确版本
cuda、cudnn安装
cuda安装

https://developer.nvidia.com/cuda-toolkit-archive

conda 版本选择
https://pytorch.org/get-started/previous-versions/#commands-for-versions--100