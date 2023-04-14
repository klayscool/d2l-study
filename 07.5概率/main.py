import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6])
print(multinomial.Multinomial(1, fair_probs).sample())

print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(1000, fair_probs).sample()
print('counts:', counts)
print(counts / 1000)