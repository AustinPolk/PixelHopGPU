from numba import cuda
import numpy as np
import math

'''

## original function

    for i in range(dilate, feature.shape[0]-dilate):
        for j in range(dilate, feature.shape[1]-dilate):
            tmp = []
            for ii in idx:
                for jj in idx:
                    iii = i+ii*dilate
                    jjj = j+jj*dilate
                    tmpg = feature[iii, jjj]
                    tmp.append(tmpg)
            tmp = np.array(tmp)
            tmp = np.moveaxis(tmp,0,1)
            tmp = tmp.reshape(S[0], -1)
            res[i-dilate, j-dilate] = tmp

## simplified version of original function, now this needs to be put on the gpu

    for i in range(dilate, feature.shape[0] - dilate):
        for j in range(dilate, feature.shape[1] - dilate):
            for a in range(0, feature.shape[2]):
                for b in range (0, 9):      ## len idx squared
                    for k in range(0, feature.shape[3]):
                        res[i - dilate, j - dilate, a, feature.shape[3] * b + k] = feature[i + (b//3 - 1) * dilate, j + (b%3 - 1) * dilate, a, k]

'''

def launchGPUResKernel(feature, res, dilate):
    
    global totalThreads

    fShape = feature.shape
    
    ## setup for kernel execution
    threadDimensions = np.array([(fShape[0] - 2 * dilate), (fShape[1] - 2 * dilate), fShape[2], 9, fShape[3]])
    totalThreads = threadDimensions.prod()
    threadsPerBlock = (128)
    blocksPerGrid = math.ceil(totalThreads / threadsPerBlock)

    ## transfer data to device
    d_feature = cuda.to_device(np.ascontiguousarray(feature))
    d_res = cuda.to_device(np.ascontiguousarray(res))
    d_threadDimensions = cuda.to_device(np.ascontiguousarray(threadDimensions))

    ## run GPU kernel
    GPUResKernel[blocksPerGrid, threadsPerBlock](d_feature, d_res, dilate, fShape[3], d_threadDimensions)

    ## transfer result back
    d_res.copy_to_host(res)
    
    return res

@cuda.jit
def GPUResKernel(d_feature, d_res, dilate, f3, d_threadDimensions):
    threadIdx = cuda.grid(1)

    if threadIdx < totalThreads:
        i, j, a, b, k = getIJABK(threadIdx, d_threadDimensions)
        d_res[i, j, a, f3 * b + k] = d_feature[i + (b//3) * dilate, j + (b%3) * dilate, a, k]

### get loop index parameters
@cuda.jit(device=True)
def getIJABK(m, threadDimensions):
    i = m // (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4])
    m -= i * (threadDimensions[1] * threadDimensions[2] * threadDimensions[3] * threadDimensions[4])
    j = m // (threadDimensions[2] * threadDimensions[3] * threadDimensions[4])
    m -= j * (threadDimensions[2] * threadDimensions[3] * threadDimensions[4])
    a = m // (threadDimensions[3] * threadDimensions[4])
    m -= a * (threadDimensions[3] * threadDimensions[4])
    b = m // (threadDimensions[4])
    m -= b * (threadDimensions[4])
    k = m
    return i, j, a, b, k
