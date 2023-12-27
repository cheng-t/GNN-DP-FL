import numpy as np
for i in range(10):
    tmp = np.where(np.random.rand(6272) < (64 / 6272))
    indices = np.where(np.random.rand(6272) < (64 / 6272))[0]
    print('size:',indices.size)
    print(indices)
    print(type(indices))