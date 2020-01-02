

import torch.nn as nn
import torch
 # pool of size=3, stride=2
m = nn.MaxPool1d(2, stride=1)
input = torch.randn(20, 3, 3)
output = m(input)
#print(output.size())

mask = [1,1,1,1,2,2,2,2,3,3,3,3,3]
mask = torch.tensor(mask).long().unsqueeze(0)
mask_embedding = nn.Embedding(4, 3)
mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
print(mask)
print(mask.size())
print(mask_embedding(mask))
print(mask_embedding(mask).size())
mask = 1 - mask_embedding(mask).transpose(1, 2)  # (B, L) -> (B, L, 3) -> (B, 3, L)
print(mask)
print(mask[:,0:1,:])

print(torch.__version__)