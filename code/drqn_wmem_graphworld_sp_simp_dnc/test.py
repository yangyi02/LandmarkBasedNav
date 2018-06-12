import numpy as np
import copy

L = np.zeros([6,6])
L[0,1] = 1
L[0,2] = 1
L[0,5] = 1
L[2,3] = 1
L[2,5] = 1
L[3,4] = 1
L[4,5] = 1

L = np.maximum(L, L.transpose())
L = (L+np.random.random([6,6]))/2

goal = 0

R = np.zeros_like(L)
R[:,goal] = L[:,goal]

V = np.zeros(L.shape[0])
gamma = 0.5

for i in range(1000):
    Q = (R+gamma*V)*L
    V = np.max(Q,1)
    if i==100:
        print(Q)

print(Q)