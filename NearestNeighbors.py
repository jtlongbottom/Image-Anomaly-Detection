#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:19:25 2022

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
#gt = sio.loadmat('salinas_impl_gt.mat')

"""
Can change the image used by commenting out current img and undoing comment of image desired
"""
#img= data.rocket()

#img = urban1['data'][:,:,70] #has 204 bands

#img= urban2['data'][:,:,20]

#img = impl4['X'][:,:,70]

#img = impl14['X'][:,:,70]

img = field['X'][:,:,70]

plt.imshow(img)
#img = np.transpose(img)

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



#%%

#np.shape(img)[0] is rows
#np.shape(img)[1] is columns
    
for i in range (0,np.size(img)): 
    if i < np.size(img)-np.shape(img)[1]: #iterate through all pixels
        x1 [i] = x [i+np.shape(img)[1]] #below
    if ((i+1)%np.shape(img)[1])!=0: #all pixels except last column, also why its axis [1]
        x2[i] = x[i+1] #to the right
    if i > np.shape(img)[1]: 
        x3[i] = x[i-np.shape(img)[1]]
    if (i %np.shape(img)[1])!=0: #iterate through all pixels except first one
        x4[i] = x[i-1] #to the left

#%%       
X = np.concatenate((x,x1,x2,x3,x4),axis=1)

#%%


sizes = img.shape

M = X.mean(0)
M = np.reshape(M,(5,1))

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
            
"""
Below are the other versions using two versions of the exponential distance matrix that can be commented out to use, in final product I will have separate code for each
"""
         #this is the line addition for the exponential distance matrix
        #if (i>0 and j>0 and i!=j): 
             #A[i][j] = 0.5
         
         #this is code for third version
         #u1 and u3 distance
        #if (i==1 and j ==3):
             #A[i][j] = 0.5
        #if (i==3 and j==1):
             #A[i][j] = 0.5
         #u2 and u4 distance
        #if (i==2 and j ==4):
             #A[i][j] = 0.5
        #if (i==4 and j==2):
             #A[i][j] = 0.5
            
#%%

A = 1/((1+(A/a))**2)

A = A - np.identity(5)


D = np.diag(np.sum(A,axis=1))
L = D - A
Dpower = fractional_matrix_power(D,-1/2)
L = Dpower * L * Dpower
end = sizes[0] * sizes[1]

for j in range(1,end):
    x = X[j,0:]
    x = np.reshape(x,[5,1])
    b = x-M
    c = np.matmul(np.transpose(b),np.matmul(L,(b)))
    out[j] = c.real
    
out = np.reshape(out,np.shape(img))
#%%


thresh = np.mean(out)
histogram, bin_edges = np.histogram(out, bins=10, range=(0.0, 300000.0))

fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
#plt.show()

#%%
t = 20000000
binary_mask = out > t

fig, ax = plt.subplots()
plt.title(t)
plt.imshow(binary_mask, cmap="gray")
plt.show()
#%%

plt.gray()
plt.imshow(out)

plt.show()