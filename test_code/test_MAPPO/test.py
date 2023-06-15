import torch

from distributions import FixedCategorical
import numpy as np
import torch.distributions.categorical as c

a = np.random.random((2, 3))

a = torch.from_numpy(a)

print(a)

print(FixedCategorical(logits=a).sample())

print(c.Categorical(logits=a).sample())