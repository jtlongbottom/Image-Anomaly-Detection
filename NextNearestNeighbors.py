#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 07:05:31 2022

@author: jakoblongbottom
"""



import matplotlib.image as image
import numpy as np
from scipy.linalg import fractional_matrix_power
from skimage import data
from skimage import color
import matplotlib.pyplot as plt
import scipy.io as sio
urban1 = sio.loadmat('urban-1.mat') #use band 70
urban2 = sio.loadmat('urban-2.mat') #use band 20
impl4 = sio.loadmat('salinas_impl_4.mat')
impl14 = sio.loadmat('salinas_impl_14.mat') # just makes the splotches white instead of black in impl4
field = sio.loadmat('salinas.mat')
field_gt = sio.loadmat('salinas_gt.mat')
impl14_gt = sio.loadmat('salinas_impl_gt.mat')


#img= data.rocket()

img = urban1['data'][:,:,70] #has 204 bands

#img= urban2['data'][:,:,20]

#img = impl4['X'][:,:,70]

#img = impl14['X'][:,:,70]

#img = impl14_gt['gt']

#img = field['X'][:,:,70]

#img = field_gt['gt']

plt.imshow(img)

isize = np.size(img) #number of pixels 
isize = int(isize)

#%%
x = np.array([])
for i in range(0,np.shape(img)[0]):
    for j in range(0,np.shape(img)[1]):
        x = np.concatenate((x,img[i][j]), axis=None)

        
x = np.reshape(x,(isize,1))

x1 = x

#SETTING the other x arrays to be equal to x1
x2 = x +0
x3 = x+0
x4 = x+0
x5 = x+0
x6 = x+0
x7 = x+0
x8 = x+0
x9 = x+0
x10 = x+0
x11 = x+0
x12 = x+0


#%%


for i in range (0,np.size(img)): 
    if i < np.size(img)-np.shape(img)[1]: #iterate through all pixels
        x1 [i] = x [i+np.shape(img)[1]] #below
    if ((i+1)%np.shape(img)[1])!=0: #all pixels except last column, also why its axis [1]
        x2[i] = x[i+1] #to the right
    if i > np.shape(img)[1]: 
        x3[i] = x[i-np.shape(img)[1]] #up
    if (i %np.shape(img)[1])!=0: #iterate through all pixels except first one
        x4[i] = x[i-1] #to the left
        
    if i < np.size(img)-np.shape(img)[1] and ((i+1)%np.shape(img)[1])!=0:
        x5[i] = x[i+np.shape(img)[1]+1] #down and to the right
    if i > np.shape(img)[1] and ((i+1)%np.shape(img)[1])!=0:
        x6[i] = x[i-np.shape(img)[1]+1]
    if i > np.shape(img)[1] and ((i)%np.shape(img)[1])!=0:
        x7[i] = x[i-np.shape(img)[1]-1]
    if i < np.size(img)-np.shape(img)[1] and ((i)%np.shape(img)[1])!=0:
        x8[i] = x[i+np.shape(img)[1]-1]

    if i < np.size(img)-(2*np.shape(img)[1]): 
        x9 [i] = x [i+(2*np.shape(img)[1])] #down 2
    if ((i+1)%np.shape(img)[1])!=0 and ((i+2)%np.shape(img)[1])!=0: 
        x10[i] = x[i+2] #to the right 2
    if i > 2*np.shape(img)[1]: 
        x11[i] = x[i-2*np.shape(img)[1]] #up 2
    if (i %np.shape(img)[1])!=0 and (i-1 %np.shape(img)[1])!=0:
        x12[i] = x[i-2] #to the left 2




#%%       
X = np.concatenate((x, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),axis=1)

#%%


sizes = img.shape

M = X.mean(0)
M = np.reshape(M,(np.shape(X)[1],1))

out = np.zeros([isize])

Mt = np.transpose(M)

A = abs(M-Mt)

a = np.mean(M)
a = np.reshape(a,(1,1))

#%%

for i in range(0,A.shape[0]):
    for j in range (0,A.shape[1]): 
        if (i==j):
            continue
        
        if (i!=j):
            A[i][j] = 1
        if i == 0 and j>8:
            A[i][j] = 1/2
        if i == 1:
            if j == (3 or 6 or 7 or 10 or 12):
                A[i][j] = 1/2
            if j==11:
                A[i][j] = 1/4
        if i == 2:
            if j == (4 or 7 or 8 or 9 or 11):
                A[i][j] = 1/2
            if j==12:
                A[i][j] = 1/4
        if i == 3:
            if j == (1 or 5 or 8 or 10 or 12):
                A[i][j] = 1/2
            if j==9:
                A[i][j] = 1/4
        if i == 4:
            if j == (2 or 5 or 6  or 9 or 11):
                A[i][j] = 1/2
            if j==10:
                A[i][j] = 1/4
        if i == 5:
            if j == (3 or 4 or 6 or 8):
                A[i][j] = 1/2
            if j==(7 or 11 or 12):
                A[i][j] = 1/4
        if i == 6:
            if j == (1 or 4 or 5 or 7):
                A[i][j] = 1/2
            if j==(8 or 9 or 12):
                A[i][j] = 1/4
        if i == 7:
            if j == (1 or 2 or 6 or 8):
                A[i][j] = 1/2
            if j==(5 or 9 or 10):
                A[i][j] = 1/4
        if i == 8:
            if j == (2 or 3 or 5 or 7):
                A[i][j] = 1/2
            if j==(6 or 10 or 11):
                A[i][j] = 1/4
        if i == 9:
            if j == (0 or 2 or 4):
                A[i][j] = 1/2
            if j==(3 or 6 or 7 or 10 or 12):
                A[i][j] = 1/4
            if j== 11:
                A[i][j] = 1/8
        if i == 10:
            if j == (0 or 1 or 4):
                A[i][j] = 1/2
            if j==(5 or 7 or 8 or 9 or 11):
                A[i][j] = 1/4
            if j== 12:
                A[i][j] = 1/8
        if i == 11:
            if j == (0 or 2 or 4):
                A[i][j] = 1/2
            if j==(1 or 5 or 8 or 10 or 12):
                A[i][j] = 1/4
            if j== 9:
                A[i][j] = 1/8
        if i == 12:
            if j == (0 or 1 or 3):
                A[i][j] = 1/2
            if j==(2 or 5 or 6 or 9 or 11):
                A[i][j] = 1/4
            if j== 10:
                A[i][j] = 1/8
        
            
#%%

A = 1/((1+(A/a))**2)


A = A - np.identity(np.shape(X)[1])


D = np.diag(np.sum(A,axis=1))
L = D - A
Dpower = fractional_matrix_power(D,-1/2)
L = Dpower * L * Dpower
end = sizes[0] * sizes[1]

for j in range(1,end):
    x = X[j,0:]
    x = np.reshape(x,[np.shape(X)[1],1])
    b = x-M
    c = np.matmul(np.transpose(b),np.matmul(L,(b)))
    out[j] = c.real
    
out = np.reshape(out,np.shape(img))



plt.gray()
plt.imshow(out)

plt.show()