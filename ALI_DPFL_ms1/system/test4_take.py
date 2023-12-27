import numpy as np
from mindspore import Tensor
a = Tensor(np.array([4, 3, 5, 7, 6, 8]))
indices = Tensor(np.array([0, 1, 4]))
output = a.take(indices)
print(output)

