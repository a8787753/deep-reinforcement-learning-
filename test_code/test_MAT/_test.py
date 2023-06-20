import numpy as np
import torch

# batch_size = 6
#
# num_mini_batch = 2
#
# mini_batch_size = batch_size // num_mini_batch
#
# num_agents = 3
#
#
# rows = np.indices((batch_size, num_agents))[0]
#
# cols = np.stack([np.arange(num_agents) for _ in range(batch_size)])
#
# ibs = np.random.random((3, 3, num_agents, 1))
#
# ibs_r = ibs.reshape(-1, *ibs.shape[2:])
#
# ibs_cr = ibs_r[rows, cols]
#
# rand = torch.randperm(batch_size).numpy()
#
# sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

a = torch.from_numpy(np.random.random((2,2,2)))
print(a)

n = a.view(-1, 2)

print(n)

m = a.reshape(-1, 2)
print(m)

