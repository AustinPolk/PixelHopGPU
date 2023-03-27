import numpy as np 
from numba import cuda
import math

''' function that was converted to gpu kernel

for i in range(dilate, feature.shape[0]-dilate):        ## range of size (feature.shape[0] - 2 * dilate)
    #
        for j in range(dilate, feature.shape[1]-dilate):    ## range of size (feature.shape[1] - 2 * dilate). Total range comes to (feature.shape[0] - 2 * dilate) * (feature.shape[1] - 2 * dilate) iterations, in two dimensions
        #
            tmp = []                                        ## tmp does not need to be reallocated each time, its data can simply be overwritten
            for ii in idx:                                  ## loops for a total of dimensions(idx) = 3 times
            #
                for jj in idx:                              ## loops for a total of dimensions(idx) = 3 times
                #
                    iii = i+ii*dilate
                    jjj = j+jj*dilate
                    tmp.append(feature[iii, jjj])           ## No need to use an append function since we can allocate tmp ahead of time and access by index directly. Index in tmp appears to be ii * 3 + jj, and tmp final size looks like 3 * 3 = 9
                #
            #
            tmp = np.array(tmp)                             ## No need to do this, as tmp should always be a numpy array
            tmp = np.moveaxis(tmp,0,1)                      
            res[i-dilate, j-dilate] = tmp.reshape(S[0],-1)  ## reshape the array into a matrix of dimensions S[0] and len(tmp) / S[0] (presumably, based on how reshape works). I believe if we can know the size of tmp beforehand, we can just directly replace the -1 (probably not that useful)
        #
    #

'''

def launchGPUResKernel(flatFeature, featureShape, flatRes, resShape, dilate):
    
    global fShape, rShape, threadDimensions

    fShape = featureShape
    rShape = resShape
    
    threadDimensions = (rShape[0], rShape[1], 3, 3, fShape[2], fShape[3])
    totalThreads = np.asarray(threadDimensions).prod()
    
    threadsPerBlock = (32)
    blocksPerGrid = math.ceil(totalThreads / threadsPerBlock)
    
    ## TODO: suppress warnings regarding memory overhead from copying
    GPUResKernel[blocksPerGrid, threadsPerBlock](flatFeature, flatRes, dilate)
    
    res = flatRes.reshape(rShape)               ## unflatten res
   
    return res

@cuda.jit
def GPUResKernel(d_feature, d_res, dilate):
    ## i and j correlate to iteration number in original loop, ii and jj correlate to iteration number in idx loops, and fi and fj correlate to coordinates of elements in the matrices produced by feature[x, y]
    
    threadIdx = cuda.grid(1)        ## index in flattened 6D thread space
    fj = (threadIdx) % threadDimensions[5]  ## position in 6th dimension
    threadIdx //= threadDimensions[5]       ## remove 6th dimension
    fi = (threadIdx) % threadDimensions[4]  ## position in 5th dimension
    threadIdx //= threadDimensions[4]       ## remove 5th dimension
    jj = (threadIdx) % threadDimensions[3]  ## position in 4th dimension
    threadIdx //= threadDimensions[3]       ## remove 4th dimension
    ii = (threadIdx) % threadDimensions[2]  ## position in 3th dimension
    threadIdx //= threadDimensions[2]       ## remove 3rd dimension
    j = (threadIdx) % threadDimensions[1]   ## position in 2th dimension
    threadIdx //= threadDimensions[1]       ## remove 2nd dimension
    i = (threadIdx)                         ## position in 1st dimension

    if i < threadDimensions[0]:     
        ## assign from feature to res
        tmpf = Access4D(d_feature, i+f_idx(ii)*dilate+dilate, j+f_idx(jj)*dilate+dilate, fi, fj)
        Assign4D(d_res, i, j, fi, fj, tmpf)

## access the flattened feature array in 4D
@cuda.jit(device=True)
def Access4D(arr, w, x, y, z):
    return arr[fShape[3] * fShape[2] * fShape[1] * w + fShape[3] * fShape[2] * x + fShape[3] * y + z]

## access the flattened res array in 4D
@cuda.jit(device=True)
def Assign4D(arr, w, x, y, z, value):
    arr[rShape[3] * rShape[2] * rShape[1] * w + rShape[3] * rShape[2] * x + rShape[3] * y + z] = value

## function that mimics an access to the idx array
@cuda.jit(device=True)
def f_idx(i):
    return i - 1