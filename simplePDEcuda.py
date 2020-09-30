# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:26:26 2020


#!/usr/local/anaconda3/bin/python3

from numpy import *
from numpy.linalg import norm
import time
from numba import cuda

@cuda.jit('void(float64[:],float64[:],float64)')
def heat(y0,y1,c):
    nPts = len(y0)
    j = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    if j==0 or j>=nPts-1:
       y0[j] = 0.
       y1[j] = 0.
       return
    y1[j] = (1-2.*c)*y0[j] + c*(y0[j-1]+y0[j+1])

nPts = 150
T = 1
h = 1./(nPts-1)
dT = h*h/4.
x = arange(0,1+h,h)

yInit = sin(2*pi*x)*cos(2*pi*x)*exp(sin(pi*x))+5
ySoln = exp(-.4*pi*pi*1)*sin(2*pi*x)
yEnd = 0.*yInit

blockDim = 64
gridDim = int(ceil(nPts/blockDim))
yInit = sin(2*pi*x)
yEnd = 0.*yInit
t0 = time.time()
nSteps = int(T/dT)
c = .1*dT/(h*h)
dy1 = cuda.to_device(yInit)
dy2 = cuda.to_device(yEnd)
for i in range(nSteps):
   heat[gridDim,blockDim](dy1,dy2,c)
   tmp = dy1
   dy1 = dy2
   dy2 = tmp
dy1.copy_to_host(yInit)
elapsedTime = time.time()-t0
print('elapsed time is {}'.format(elapsedTime))
print('error was {}'.format(norm(yInit-ySoln)/norm(yInit)))

#plot(x,yInit)
#show()
