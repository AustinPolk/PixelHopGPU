import numpy as np 
from numba import cuda
import math

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
    
    res = flatRes.reshape(rShape)
   
    return res

@cuda.jit
def GPUResKernel(d_feature, d_res, dilate):
    ## i and j correlate to iteration number in original loop, ii and jj correlate to iteration number in idx loops, and fi and fj correlate to coordinates of elements in the matrices produced by feature[x, y]
    
    threadIdx = cuda.grid(1)                ## 6D index
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
        tmpf = AccessFeature(d_feature, i+ii*dilate, j+jj*dilate, fi, fj)
        AssignRes(d_res, i, j, fi, fj, tmpf)

## access the flattened feature array in 4D
@cuda.jit(device=True)
def AccessFeature(arr, w, x, y, z):
    return arr[fShape[3] * fShape[2] * fShape[1] * w + fShape[3] * fShape[2] * x + fShape[3] * y + z]

## access the flattened res array in 4D
@cuda.jit(device=True)
def AssignRes(arr, w, x, y, z, value):
    arr[rShape[3] * rShape[2] * rShape[1] * w + rShape[3] * rShape[2] * x + rShape[3] * y + z] = value