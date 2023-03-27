from numba import cuda
import math

def launchGPUResKernel(flatFeature, featureShape, flatRes, resShape, dilate):
    
    global fShape, rShape, threadDimensions

    fShape = featureShape
    rShape = resShape
    
    threadDimensions = (fShape[0] - 2 * dilate, fShape[1] - 2 * dilate, rShape[2] * rShape[3])
    
    threadsPerBlock = (6, 6, 6)
    blocksPerGrid_x = math.ceil(threadDimensions[0] / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(threadDimensions[1] / threadsPerBlock[1])
    blocksPerGrid_z = math.ceil(threadDimensions[2] / threadsPerBlock[2])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y, blocksPerGrid_z)

    ## TODO: suppress warnings regarding memory overhead from copying
    GPUResKernel[blocksPerGrid, threadsPerBlock](flatFeature, flatRes, dilate)
    
    res = flatRes.reshape(rShape)
   
    return res

@cuda.jit
def GPUResKernel(d_feature, d_res, dilate):
    i, j, r = cuda.grid(3)

    if i < threadDimensions[0] and j < threadDimensions[1] and r < threadDimensions[2]:     
        rj = r % rShape[3]
        ri = r // rShape[3]

        iii = i + (ri % 3) * dilate         ## might need to mod something else by 3
        jjj = j + (rj % 3) * dilate

        fj = rj
        fi = (9 * ri) % fShape[2]

        tmpf = AccessFeature(d_feature, iii, jjj, fi, fj)
        AssignRes(d_res, i, j, ri, rj, tmpf)

## access the flattened feature array in 4D
@cuda.jit(device=True)
def AccessFeature(arr, w, x, y, z):
    return arr[fShape[3] * fShape[2] * fShape[1] * w + fShape[3] * fShape[2] * x + fShape[3] * y + z]

## access the flattened res array in 4D
@cuda.jit(device=True)
def AssignRes(arr, w, x, y, z, value):
    arr[rShape[3] * rShape[2] * rShape[1] * w + rShape[3] * rShape[2] * x + rShape[3] * y + z] = value