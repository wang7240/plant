# data_transforms.py
import numpy as np
import torch
from torchvision.transforms import Resize

class ToTensorSafe:
    """
    Resize 后把 PIL.Image 可靠地转成 float32 Tensor(C,H,W)，值归一到 [0,1]。
    支持灰度图和 RGB。
    """
    def __init__(self, size):
        self.resize = Resize(size)

    def __call__(self, pic):
        # 1. Resize（结果仍然是 PIL.Image）
        pic = self.resize(pic)
        # 2. 转 numpy.ndarray，dtype=np.uint8
        arr = np.array(pic, copy=True, dtype=np.uint8)
        # 3. 灰度图补通道
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # 4. HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        # 5. 转 torch Tensor 并归一到 [0,1]
        return torch.tensor(arr, dtype=torch.float32).div(255.0)
