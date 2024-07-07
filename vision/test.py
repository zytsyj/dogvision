import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 显示当前CUDA版本
print(torch.version.cuda)