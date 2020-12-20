import torch
import time

n = 40000
loop = 1000

###CPU
start_time = time.time()
a = torch.ones(n,n)
for _ in range(loop):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(n,n).cuda()
for _ in range(loop):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)