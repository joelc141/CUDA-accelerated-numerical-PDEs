# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:36:12 2020

@author: joelc
"""
from scipy import linalg
import numpy as np
import scipy as sci
import numba as cuda
import cupy as cupy
import time

m=320 #steps in x
n=200 #steps in time
A=cupy.zeros((m,m))
Asol=cupy.zeros((m,n))
B=cupy.random.rand(m)
#BB=np.random.rand(500)

a=0 #left endpoint
b=1 #right endpoint
dx=(b-a)/m
k=3  #heat constant


for i in range(1,m-1):
    A[i,i-1]=-k
    A[i,i]=(1+2*k)
    A[i,i+1]=-k

A[0,0]=1;
A[m-1,m-1]=1;
A=1/dx^2*A
t0 = time.time()
bsol=cupy.linalg.solve(A,B)
t1 = time.time()
total=t1-t0

AA=cupy.asnumpy(A)
BB=cupy.asnumpy(B)

t00=time.time()
np.linalg.solve(AA,BB)
t001=time.time()

total2=t001-t00

x=cupy.linspace(0,100,m)

xx=cupy.sin(x)
Asol[:,1]=xx
x=xx
t0matrix=time.time()
A=cupy.linalg.inv(A)
Asol[:,0]=x
for i in range(1,n):
     A=cupy.matmul(A,A)
     Asol[:,i]=cupy.matmul(A,x)

tfmatrix=time.time()

GPUtime=tfmatrix-t0matrix
AAsol=cupy.asnumpy(Asol)

x=cupy.asnumpy(xx)
t0matrixnumpy=time.time()
AAsol[:,0]=x
#for i in range(1,n):
    #AAsol[:,i]=np.linalg.solve(AA,AAsol[:,i-1])
    
for i in range(1,n):
    AAsol[:,i]=np.matmul(np.linalg.matrix_power(AA,i),x)
        
tfmatrixnumpy=time.time()

nptime=tfmatrixnumpy-t0matrixnumpy




