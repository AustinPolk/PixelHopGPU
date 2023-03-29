from numba import cuda
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

def launchGPUResKernel(flatFeature, featureShape, flatRes, resShape, dilate):
    
    global fShape, rShape, threadDimensions

    fShape = featureShape
    rShape = resShape
    
    threadDimensions = (fShape[0] - 2 * dilate, fShape[1] - 2 * dilate, fShape[2] * 9 * fShape[3])
    
    threadsPerBlock = (6, 6, 6)
    blocksPerGrid_x = math.ceil(threadDimensions[0] / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(threadDimensions[1] / threadsPerBlock[1])
    blocksPerGrid_z = math.ceil(threadDimensions[2] / threadsPerBlock[2])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y, blocksPerGrid_z)

    d_feature = cuda.to_device(flatFeature)
    d_res = cuda.to_device(flatRes)

    GPUResKernel[blocksPerGrid, threadsPerBlock](d_feature, d_res, dilate)

    d_res.copy_to_host(flatRes)

    res = flatRes.reshape(rShape)
   
    return res

@cuda.jit
def GPUResKernel(d_feature, d_res, dilate):
    i, j, m = cuda.grid(3)

    if i < threadDimensions[0] and j < threadDimensions[1] and m < threadDimensions[2]:
        i += dilate
        j += dilate
        
        k = m % fShape[3]       ## get k loop param
        m //= fShape[3]
        b = m % 9               ## get b loop param
        m //= 9
        a = m                   ## get a loop param

        tmpf = AccessFeature(d_feature, i + (b//3 - 1) * dilate, j + (b%3 - 1) * dilate, a, k)
        AssignRes(d_res, i - dilate, j - dilate, a, fShape[3] * b + k, tmpf)


## access the flattened feature array in 4D
@cuda.jit(device=True)
def AccessFeature(arr, w, x, y, z):
    return arr[fShape[3] * fShape[2] * fShape[1] * w + fShape[3] * fShape[2] * x + fShape[3] * y + z]

## access the flattened res array in 4D
@cuda.jit(device=True)
def AssignRes(arr, w, x, y, z, value):
    arr[rShape[3] * rShape[2] * rShape[1] * w + rShape[3] * rShape[2] * x + rShape[3] * y + z] = value