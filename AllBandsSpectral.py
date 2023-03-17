#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:19:25 2022

@author: jakoblongbottom
"""



import matplotlib.image as image1
import numpy as np
from scipy.linalg import fractional_matrix_power
from skimage import data
from skimage import color
import matplotlib.pyplot as plt
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


isize = np.size(img[0][0][:]) #number of pixels 
isize = int(isize)


x = np.array([])


#%%


sizes = img.shape
bands = sizes[2]
X = np.reshape(img,[sizes[0]*sizes[1],bands])

M = X.mean(0)
M = np.reshape(M,(bands,1))

out = np.zeros([sizes[0]*sizes[1]])


Mt = np.transpose(M)

A = abs(M-Mt)

a = np.mean(M)
a = np.reshape(a,(1,1))

#%%

A = 1/((1+(A/a))**2)

A = A - np.identity(bands) 


D = np.diag(np.sum(A,axis=1))
L = D - A
Dpower = fractional_matrix_power(D,-1/2)
L = Dpower * L * Dpower
end = sizes[0] * sizes[1]

for j in range(1,end): 
    x = X[j,:]
    x = np.reshape(x,[bands,1]) 
    b = x-M
    c = np.matmul(np.transpose(b),np.matmul(L,(b))) 
    out[j] = c.real
    
out = np.reshape(out, [sizes[0],sizes[1]])
#%%
max = np.max(out)
#11.7 million

out = out/ max


histogram, bin_edges = np.histogram(out, bins=10, range=(0.0, 1.0))


#%%
max_thresh_SOI = 0
max_thresh = 0

for i in range(1,100):
    thresh = i/100.0
    binary_mask = out > thresh
    binary_mask = binary_mask*1

    thresh_num = 0
    thresh_denom = 0

    for i in range(0,len(binary_mask)):
        for j in range(0,len(binary_mask[0])):
            if (binary_mask[i][j] == 1 and ground[i][j]==1):
                thresh_num =thresh_num+2
                thresh_denom =thresh_denom+2
            elif (binary_mask[i][j]==1 or ground[i][j] == 1):
                thresh_denom =thresh_denom+1
            
    thresh_SOI = (thresh_num)/thresh_denom

    if (thresh_SOI > max_thresh_SOI):
        max_thresh_SOI = thresh_SOI
        max_thresh = thresh

print("Spectral Threshold SOI for " + image)
print(np.round(max_thresh_SOI,4))
print( " at ")
print(max_thresh)  

binary_mask = out > max_thresh
    
fig, ax = plt.subplots()
plt.title("{image} at t={max_thresh}".format(image = image, max_thresh = max_thresh))
plt.imshow(binary_mask, cmap="gray")
plt.show()

#%%


