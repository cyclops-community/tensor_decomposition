import ctf
import numpy as np
import time
import backend.ctf_ext as backctf
import backend.numpy_ext as backnp

A = []
size =500
for i in range(3):
    A.append(ctf.random.random((500,500)))

C= A.copy()    
    
start=  time.time()

res = backctf.list_add(A,C)
end = time.time()

print('time for ctf in list add:',end- start)


start=  time.time()

res = backctf.scalar_mul(2,C)
end = time.time()

print('time for ctf in scalar mul:',end- start)

B = []

for i in range(3):
    B.append(np.random.rand(500,500))
    
D = B.copy()


start=  time.time()

res = backnp.list_add(B,D)
end = time.time()

print('time for numpy in list add:',end- start)


start=  time.time()

res = backnp.scalar_mul(2,B)
end = time.time()

print('time for numpy in scalar mul:',end- start)

