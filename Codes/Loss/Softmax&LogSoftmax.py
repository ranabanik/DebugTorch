import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

torch.manual_seed(1001)

y = torch.randn(1, 10)
"""lets make one element 0"""

y[0, 4] = 0.2
# y = y.squeeze(0) # just to plot reducing the dimension
print(y.shape)

sm = nn.Softmax(dim=1)
lsm = nn.LogSoftmax(dim=1)

ysm = sm(y).squeeze(0)
ylsm = lsm(y).squeeze(0)

fig, ax = plt.subplots(figsize=(10, 10))
x = torch.arange(10).squeeze(0)
ax.plot(x, y.squeeze(0), 'g', label='Input')
ax.plot(x, ysm, 'r', label='Softmax')
ax.plot(x, ylsm, 'b', label='LogSoftmax')
ax.axis('equal')
ax.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title()
leg = ax.legend(fontsize=20)
# plt.show()

imgDir = r'C:\Users\ranab\PycharmProjects\DebugTorch\Images'
plt.savefig(os.path.join(imgDir, 'softlogmax.png'))

plt.show()