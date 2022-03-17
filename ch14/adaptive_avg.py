
import torch

# (batch, channels, H, W)
img = torch.randn(1, 2, 4, 4)

batch_norm = torch.nn.BatchNorm2d(2)
adaptive_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

# (batch, channels, H, W)
tmp = batch_norm(img)

print(tmp)
# output is a function of the shape
print(adaptive_pool(tmp).shape)