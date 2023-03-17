#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:09:34 2023

@author: jakoblongbottom
"""

import matplotlib.image as image
import numpy as np
from scipy.linalg import fractional_matrix_power
from skimage import data
from skimage import color
import matplotlib.pyplot as plt
import numpy.matlib
import scipy.io as sio
urban1 = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/urban-1.mat') #use band 70
urban2 = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/urban-2.mat') #use band 20
impl4 = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/salinas_impl_4.mat')
impl14 = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/salinas_impl_14.mat') # just makes the splotches white instead of black in impl4
field = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/salinas.mat')
fieldgt = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/salinas_gt.mat')
implgt = sio.loadmat('/Users/jakoblongbottom/Desktop/Phys 489/Phys 489 thesis/salinas_impl_gt.mat')

"""
Can change the image used by commenting out current img and undoing comment of image desired
"""
#img= data.rocket()


img = urban1['data'] #has 204 bands, use 70
imgband = 70
image = "Urban-A"
ground=urban1['map']


"""
img= urban2['data'] #use 20 band
imgband = 20
image = "Urban-B"
ground = urban2['map']
"""

"""
img = impl4['X']
imgband = 70
image = "Impl-4"
ground = implgt['gt']
"""


"""
img = impl14['X']
imgband = 70
image = "Impl-14"
ground = implgt['gt']
"""

"""
img = field['X']
imgband = 70
image = "Field"
ground = fieldgt['gt']
"""

plt.title("Band {imgband} of {image}".format(imgband = imgband, image = image))
plt.imshow(img[:,:,imgband])
plt.show()


plt.title(image + ' Ground Truth')
plt.imshow(ground)
#img = np.transpose(img)

isize = np.size(img[0][0][:]) #number of pixels 
isize = int(isize)


#%%

sizes = img.shape
bands = sizes[2]
pixels = sizes[0]*sizes[1]
X = np.reshape(img,[pixels,bands])








i = np.reshape(np.array(range(0,pixels)),[pixels,1])
ii = np.matlib.repmat(i,1,5)

nb = np.reshape(np.array([0,1,-1,sizes[0],-sizes[0]]),[1,5]) #x, right, left, down, up
nbb = np.matlib.repmat(nb,pixels,1)

what = ii+nbb

for k in range(0,pixels):
    if (k % sizes[1] == sizes[1]-1):
        what[k][1] = ii[k][1] #what[k][1] -1
        
    if ((k % sizes[1]) == 0):
        what[k][2] = ii[k][2] #what[k][2]+1
        
    if (k >= (pixels-sizes[0])):        #9900:
        what[k][3] = ii[k][3]
        
    if (k < sizes[0]):
        what[k][4] = ii[k][4]


    
 
    
ii = np.kron(what,np.ones(bands))


j = np.reshape(np.array(range(bands)),[1,bands])
j=(j) * pixels

jj = np.matlib.repmat(j,pixels,5) #this is correct
ij=ii+jj


x = ii


#np.shape(ij)[0]  = pixels
#np.shape(ij)[1] = 1020 (5*bands)


for y in range(np.shape(x)[0]): #pixels
    for z in range(np.shape(x)[1]): #bands
        band_index = int(ij[y][z]//pixels) #could also do z%bands
        pixel_index = int(ij[y][z])%pixels
        x[y][z] = X[pixel_index][band_index]


#%%

out = np.zeros([sizes[0]*sizes[1]])

M = x.mean(0)
M = np.reshape(M,(1,5*bands))

x = x - np.matlib.repmat(M,pixels,1)

Mt = np.transpose(M)

A = abs(M-Mt)

a = np.mean(M)
a = np.reshape(a,(1,1))

            
#%%

A = 1/((1+(A/a))**2) #wcauchy function


#%%
n = 5*bands
W = np.zeros((n,n))
A1 = np.ones((bands,bands))
A2 = np.identity(bands)/2
A3 = A2/2
W[0:bands,0:n] = np.matlib.repmat(A2,1,5)
W[0:n,0:bands] = np.matlib.repmat(A2,5,1)

W[0:bands,0:bands] = A1 #x,x
W[(bands):(2*bands), (bands):(2*bands)] = A1 #1,1
W[(2*bands):(3*bands), (2*bands):(3*bands)] = A1 #2,2
W[(3*bands):(4*bands), (3*bands):(4*bands)] = A1 #3,3
W[(4*bands):(5*bands), (4*bands):(5*bands)] = A1 #4,4



W[(bands):(2*bands), (2*bands):(3*bands)] = A3 #1,2
W[(bands):(2*bands), (3*bands):(4*bands)] = A3 #1,3
W[(bands):(2*bands), (4*bands):(5*bands)] = A3 #1,4

W[(2*bands):(3*bands), (bands):(2*bands)] = A3 #2,1
W[(2*bands):(3*bands), (3*bands):(4*bands)] = A3 #2,3
W[(2*bands):(3*bands), (4*bands):(5*bands)] = A3 #2,4

W[(3*bands):(4*bands), (bands):(2*bands)] = A3 #3,1
W[(3*bands):(4*bands), (2*bands):(3*bands)] = A3 #3,2
W[(3*bands):(4*bands), (4*bands):(5*bands)] = A3 #3,4

W[(4*bands):(5*bands), (bands):(2*bands)] = A3 #4,1
W[(4*bands):(5*bands), (2*bands):(3*bands)] = A3 #4,2
W[(4*bands):(5*bands), (3*bands):(4*bands)] = A3 #4,3


A = A*W



#%%

A = A - np.identity(5*bands) #no self loops


D = np.diag(np.sum(A,axis=1)) #diagonal matrix of degrees
L = D - A #Laplacian matrix, should be a 1020x1020
Dpower = fractional_matrix_power(D,-1/2)
L = Dpower * L * Dpower #normalized laplacian matrix


for k in range(0,pixels):
    Xx = x[k,:] #all the band values of a single pixel
    Xx = np.reshape(Xx,[5*bands,1]) # simply getting into column array
    b = Xx
    bt = np.transpose(b)
    c = np.matmul(bt,np.matmul(L,(b)))
    pixel_num = i[k]
    out[pixel_num] = c.real
    
out = np.reshape(out, [sizes[0],sizes[1]])



#%%
max = np.max(out)
#11.7 million

out = out/ max

#%%
max_thresh_SOI = 0
max_thresh = 0

for k in range(1,100):
    thresh = k/100.0
    binary_mask = out > thresh
    binary_mask = binary_mask*1

    thresh_num = 0
    thresh_denom = 0

    for i in range(0,len(binary_mask)):
        for j in range(0,len(binary_mask[0])):
            if (binary_mask[i][j] == 1 and ground[i][j]==1): #true positive
                thresh_num =thresh_num+2
                thresh_denom =thresh_denom+2
            elif (binary_mask[i][j]==1 or ground[i][j] == 1): #false positive and false negative
                thresh_denom =thresh_denom+1
            
    thresh_SOI = (thresh_num)/thresh_denom

    if (thresh_SOI > max_thresh_SOI):
        max_thresh_SOI = thresh_SOI
        max_thresh = thresh

print("Exp Dist SOI for " + image)
print(np.round(max_thresh_SOI,4))
print( " at threshhold of ")
print(max_thresh)  

binary_mask = out > max_thresh
    
fig, ax = plt.subplots()
plt.title("{image} at t={max_thresh}".format(image = image, max_thresh = max_thresh))
plt.imshow(binary_mask, cmap="gray")
plt.show()

