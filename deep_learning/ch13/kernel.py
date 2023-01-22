
import torch
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('data/img_10.jpg')
img = np.asarray(img) / 255.
# size = (batch, color, H, W)
img = torch.from_numpy(img).view(1, 1, 28, 28).float()

kernel = 5
layer = torch.nn.Conv2d(in_channels=1, out_channels=kernel,
                        kernel_size=(3, 3), stride=(1, 1), padding=0)

# size = (1, kernel, H, W)
conv = layer(img).detach().numpy()

print(layer.weight)
print(conv.shape)

# print convolutions
fig = plt.figure()
for i in range(kernel):
    fig.add_subplot(1, kernel, i + 1)
    plt.imshow(conv[0][i])
plt.show()
