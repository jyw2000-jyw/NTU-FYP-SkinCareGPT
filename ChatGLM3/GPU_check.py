# !/user/bin/env python3
# -*- coding: utf-8 -*-

import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("可用的 CUDA GPU 数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA 不可用")
