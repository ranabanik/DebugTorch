import numpy as np
import h5py
import matplotlib.pyplot as plt
from BabyNetwork.ConvNetStepbyStepUtils import zero_pad

np.random.seed(1)

X = np.random.randint(0, 100, size=(5, 100, 100, 3))

# plt.imshow(X)
# plt.show()
# X_padded = np.pad(X, ((0, 0), (10, 5), (20, 10), (0, 0)), mode='constant', constant_values=(0, 0)) # works
X_padded = zero_pad(X, 5)
plt.imshow(X_padded[1, :, :, :])
plt.show()