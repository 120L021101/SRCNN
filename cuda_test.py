import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.version.cuda)  # 检查 PyTorch 绑定的 CUDA 版本
print(torch.cuda.is_available())  # 如果仍然是 False，可能是库路径问题
print(torch.cuda.device_count())  # 检查 GPU 数量
