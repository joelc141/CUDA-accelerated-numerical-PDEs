# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:46:40 2020

@author: joelc
"""
import numpy as np
import cupy as cp
import time

Nx=10
Nt=10
k=1  #dispersion parameter
b=2  #flow speed

x0=0
xf=100

t0=0
tf=1000
dx=xf/Nx
dt=(tf-t0)/Nt
A=1-2*k*dt/(dx*dx)-b*dt/dx
B=k*dt/(dx*dx)-b*dt/dx
C=k*dt/(dx*dx)


t=cp.linspace(t0,tf,Nt)
x=cp.linspace(x0,xf,Nx)


k1=C
k2=A+B


IC=0*t  #initial condition
BC=np.sin(x)  #boundary condition in time 
xx=[C,A,B]
AA=np.zeros((Nx,Nx))
#construct first row of matrix A
AA[0,:]=np.append([C,A+B],np.zeros((Nx-2,1)))
#AA[Nx,:]=
#construct diagonal elements
for i in range(1,Nx-1):
    AA[i,:]=np.append(np.append(np.zeros(i-1),xx),np.zeros(Nx-3-(i-1)))
#we now have the characteristic matrix for the convective heat equation

Ct=[] #this will be unrolled into a solution matrix, it is a "list" as of now
t1=time.time()
v=IC

b2=cp.zeros((Nx,Nx))
b2[1,:]=BC

#convert numpy to cuda matrix
AA=cp.array(AA)

for i in range(0,Nt):
    v=cp.matmul(AA,v)-C*dx*b2[:,i]
    Ct+=[v] #careful here dont change the data type otherwise it will call back to the cpu!

tf=time.time()

CUDA=t1-tf

#U_j+1=A*U_J+beta


        