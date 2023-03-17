#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:26:05 2023

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

isize = np.size(img[0][0][:]) #number of pixels 
isize = int(isize)


#%%

sizes = img.shape
bands = sizes[2]
pixels = sizes[0]*sizes[1]
X = np.reshape(img,[pixels,bands])




#%%


i = np.reshape(np.array(range(0,pixels)),[pixels,1])
ii = np.matlib.repmat(i,1,13)

nb = np.reshape(np.array([0,sizes[0],-sizes[0],1,-1,2*sizes[0],-2*sizes[0],2,-2, sizes[0]+1, -sizes[0]+1, sizes[0]-1,-sizes[0]-1]),[1,13])
 #x, down, up, right, left, 2 down, 2 up, 2 right, 2 left, downright, upright, downleft, upleft
nbb = np.matlib.repmat(nb,pixels,1)

what = ii+nbb

"""
bands = 3
pixels = 100
sizes=[10,10]
"""

for k in range(0,pixels):
    if (k >= (pixels-sizes[0])): #down
        what[k][1] = ii[k][1]
        #what[k][9] = ii[k][9]
        #what[k][11] =ii[k][11]
        
    if (k < sizes[0]):          #up
        what[k][2] = ii[k][2] 
        #what[k][10] = ii[k][10]
        #what[k][12] = ii[k][12]
        
    if (k % sizes[1] == sizes[1]-1):    #right
        what[k][3] = ii[k][3] 
        #what[k][9] = ii[k][9]
       # what[k][10] = ii[k][10]
        
    if ((k % sizes[1]) == 0):       #left
        what[k][4] = ii[k][4] #same as before inner 4 pixels
        #what[k][11] = ii[k][11]
        #what[k][12] = ii[k][12]
        
    if (k < 2*sizes[0]):        #2 up
        what[k][6] = ii[k][6]
        
    if (k >= (pixels-2*sizes[0])):  #2 down
        what[k][5] = ii[k][5]
        
    if (k%sizes[1]>= sizes[1] -2):      #2 right
        what[k][7] = ii[k][7]
        
    if (k%sizes[1] <= 1):           #2 left
        what[k][8] = ii[k][8]
        
    if (k % sizes[1]== sizes[1]-1 or k >= (pixels-sizes[0])):
        what[k][9] = ii[k][9]
        
    if (k<sizes[0] or k % sizes[1] == sizes[1]-1):
        what[k][10] = ii[k][10]
        
    if (k >= (pixels-sizes[0]) or (k % sizes[1] == 0)):
        what[k][11] = ii[k][11]
        
    if (k<sizes[0] or (k % sizes[1]) == 0):
        what[k][12] = ii[k][12]
    
    
ii = np.kron(what,np.ones(bands))


j = np.reshape(np.array(range(bands)),[1,bands])
j=(j) * pixels

jj = np.matlib.repmat(j,pixels,13) #this is correct
ij=ii+jj


#%%

x = ii


#np.shape(ij)[0]  = pixels
#np.shape(ij)[1] = 2652 (13*bands)


for y in range(np.shape(x)[0]): #pixels
    for z in range(np.shape(x)[1]): #bands
        #band_index = int(ij[y][z]//pixels) #could also do z%bands
        pixel_index = int(ij[y][z])%pixels
        x[y][z] = X[pixel_index][z%bands]


#%%

out = np.zeros([sizes[0]*sizes[1]])

M = x.mean(0)
M = np.reshape(M,(1,13*bands))

x = x - np.matlib.repmat(M,pixels,1)

Mt = np.transpose(M)

A = abs(M-Mt)

a = np.mean(M)
a = np.reshape(a,(1,1))


A = 1/((1+(A/a))**2) #wcauchy function


#%%
n = 13*bands
W = np.zeros((n,n))
A1 = np.ones((bands,bands))
A2 = np.identity(bands)/2 #1/2
A3 = A2/2 #1/4
A4 = A3/2 #1/8
A5 = A4/2 #1/16

#x-4 pixels
W[0:5*bands,0:5*bands] = np.matlib.repmat(A3,5,5)


#pixels related to x on x axis or top row
W[0:bands,bands:5*bands] = np.matlib.repmat(A2,1,4) # 1-4, 0
W[0:bands,5*bands:13*bands] = np.matlib.repmat(A3,1,8) #5-12,0

#pixels related to x on y axis or first column
W[bands:5*bands, 0: bands] = np.matlib.repmat(A2,4,1)   #0,1-4  
W[5*bands:13*bands, 0: bands] = np.matlib.repmat(A3,8,1) #0,5-12
    

#bottom left square
W[5*bands:13*bands,bands:5*bands] = np.matlib.repmat(A4,8,4) #all the 1/8s
for k in range(1,4):
     W[(k+4)*bands:(k+5)*bands,k*bands:(k+1)*bands] = A2       #5,1/ 6,2 / 7,3 / 8,4
W[9*bands:10*bands,bands:2*bands] = A2     #9,1   
W[9*bands:10*bands,3*bands:4*bands] = A2   #9,4
W[10*bands:11*bands,2*bands:4*bands] = np.matlib.repmat(A2,1,2) #10,2 and 3
W[11*bands:12*bands,bands:2*bands] = A2     #11,1   
W[12*bands:13*bands,2*bands:3*bands] = A2 #12,2
W[11*bands:13*bands,4*bands:5*bands] = np.matlib.repmat(A2,2,1) #11 and 12, 4

#top right square
W[bands:5*bands,5*bands:13*bands] = np.matlib.repmat(A4,4,8) #all the 1/8s
for k in range(1,4):
     W[(k)*bands:(k+1)*bands,(k+4)*bands:(k+5)*bands] = A2 #1,5 / 2,6 / 3,7 / 4,8
W[1*bands:2*bands,9*bands:10*bands] = A2 #1,9
W[1*bands:2*bands,11*bands:12*bands] = A2 #1,11
W[3*bands:4*bands,9*bands:10*bands] = A2 #3,9
W[4*bands:5*bands,11*bands:13*bands] = np.matlib.repmat(A2 ,1,2) #4,11 and 12
W[2*bands:4*bands,10*bands:11*bands] = np.matlib.repmat(A2 ,2,1) #2 and 3 , 10
W[2*bands:3*bands,12*bands:13*bands] = A2 #2,12


#bottom right square
W[5*bands:13*bands,5*bands:13*bands] = np.matlib.repmat(A5,8,8) #all the 1/16s
W[9*bands:10*bands,5*bands:6*bands] = A3 #9,5
W[11*bands:12*bands,5*bands:6*bands] = A3 #11,5
W[10*bands:11*bands,6*bands:7*bands] = A3 #10,6
W[12*bands:13*bands,6*bands:7*bands] = A3 #12,6
W[9*bands:11*bands,7*bands:8*bands] = np.matlib.repmat(A3,2,1) #9 and 10,7
W[11*bands:13*bands,8*bands:9*bands] = np.matlib.repmat(A3,2,1) #11 and 12,8

W[10*bands:12*bands,9*bands:10*bands] = np.matlib.repmat(A3,2,1) #10 and 11,9
W[5*bands:6*bands,9*bands:10*bands] = A3 #5,9
W[7*bands:8*bands,9*bands:10*bands] = A3 #7,9

W[6*bands:8*bands,10*bands:11*bands] = np.matlib.repmat(A3,2,1) #6 and 7,10
W[9*bands:10*bands,10*bands:11*bands] = A3 #9,10
W[12*bands:13*bands,10*bands:11*bands] = A3 #12,10

W[8*bands:10*bands,11*bands:12*bands] = np.matlib.repmat(A3,2,1) #8 and 9,11
W[5*bands:6*bands,11*bands:12*bands] = A3 #5,11
W[12*bands:13*bands,11*bands:12*bands] = A3 #12,11

W[10*bands:12*bands,12*bands:13*bands] = np.matlib.repmat(A3,2,1) #10 and 11,12
W[6*bands:7*bands,12*bands:13*bands] = A3 #6,12
W[8*bands:9*bands,12*bands:13*bands] = A3 #8,12


for k in range(0,13):
    W[k*bands:(k+1)*bands,k*bands:(k+1)*bands] = A1 #x,x

#%%

A = A*W


A = A - np.identity(13*bands) #no self loops


D = np.diag(np.sum(A,axis=1)) #diagonal matrix of degrees
L = D - A #Laplacian matrix, should be a 1020x1020
Dpower = fractional_matrix_power(D,-1/2)
L = Dpower * L * Dpower #normalized laplacian matrix



for k in range(0,pixels):
    Xx = x[k,:] #all the band values of a single pixel
    Xx = np.reshape(Xx,[13*bands,1]) # simply getting into column array
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

print("Next Nearest SOI for " + image)
print(np.round(max_thresh_SOI,4))
print( " at threshhold of ")
print(max_thresh)  

binary_mask = out > max_thresh
    
fig, ax = plt.subplots()
plt.title("{image} at t={max_thresh}".format(image = image, max_thresh = max_thresh))
plt.imshow(binary_mask, cmap="gray")
plt.show()

