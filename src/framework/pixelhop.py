# PixelHop unit

# feature: <4-D array>, (N, H, W, D)
# dilate: <int> dilate for pixelhop (default: 1)
# num_AC_kernels: <int> AC kernels used for Saab (default: 6)
# pad: <'reflect' or 'none' or 'zeros'> padding method (default: 'reflect)
# weight_name: <string> weight file (in '../weight/'+weight_name) to be saved or loaded. 
# getK: <bool> 0: using saab to get weight; 1: loaded pre-achieved weight
# useDC: <bool> add a DC kernel. 0: not use (out kernel is num_AC_kernels); 1: use (out kernel is num_AC_kernels+1)

# return <4-D array>, (N, H_new, W_new, D_new)

import math
import numpy as np 
import pickle
import time
from numba import cuda
from framework.saab import Saab

threadsPerBlock = 64        ## threads used per block for GPU Kernels

def PixelHop_Unit_GPU(feature, dilate=1, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False, energypercent=0.92):
    print("=========== Start: PixelHop_Unit_GPU")
    t0 = time.time()
    weight_path = '../weight/'+weight_name  ## weight path for saab

    ### NEIGHBOUR/BIAS PADDING ###
    print("       <Info>        dilate: %s"%str(dilate))
    print("       <Info>        padding: %s"%str(pad))
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate, dilate),(dilate, dilate),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate, dilate),(dilate, dilate),(0,0)), 'constant', constant_values=0)
    fShape = feature.shape
    resShape = (fShape[0], fShape[2] - 2*dilate, fShape[1] - 2*dilate , 9*fShape[3])

    ### NEIGHBOUR/BIAS KERNEL THREAD SETUP###
    threadDimensions = np.array([(fShape[2] - 2 * dilate), (fShape[1] - 2 * dilate), fShape[0], 9, fShape[3]])
    d_threadDimensions = cuda.to_device(np.ascontiguousarray(threadDimensions))
    totalThreads = threadDimensions.prod()
    blocksPerGrid = math.ceil(totalThreads / threadsPerBlock)

    ### TRANSFER DATA TO DEVICE ###
    d_feature = cuda.to_device(np.ascontiguousarray(feature))
    d_res = cuda.device_array(resShape)

    ### INVOKE NEIGHBOR KERNEL IF USING SAAB ###
    if getK == True:
        GPU_8_Neighbour[blocksPerGrid, threadsPerBlock](d_feature, d_res, dilate, fShape[3], d_threadDimensions)
        resNeighbour = d_res.copy_to_host()
        saab = Saab(weight_path, kernel_sizes=np.array([2]), useDC=useDC, energy_percent=energypercent)
        saab.fit(resNeighbour)

    ### GET BIAS AND WEIGHT ###
    print("       <Info>        Using weight: %s"%str(weight_path))
    fr = open(weight_path, 'rb')
    pca_params = pickle.load(fr)                
    fr.close()
    weight = pca_params['Layer_0/kernel'].astype(np.float32)
    bias = pca_params['Layer_%d/bias' % 0]
    print("       <Info>        bias value: %s"%str(bias))
    print("       <Info>        weight shape: %s"%str(weight.shape))

    ### INVOKE BIAS KERNEL ###
    GPU_Feature_Bias[blocksPerGrid, threadsPerBlock](d_feature, d_res, dilate, fShape[3], bias, d_threadDimensions)

    ### COPY RESULT AND CONTINUE ON HOST ###
    feature_w_bias = d_res.copy_to_host()
    transformed_feature = np.matmul(feature_w_bias, np.transpose(weight))
    if useDC == True:
        e = np.zeros((1, weight.shape[0]))
        e[0, 0] = 1
        transformed_feature -= bias * e
    print("       <Info>        Output feature shape: %s"%str(transformed_feature.shape))
    print("=========== End: PixelHop_Unit_GPU -> using %10f seconds"%(time.time()-t0))
    return transformed_feature

@cuda.jit
def GPU_8_Neighbour(d_feature, d_res, dilate, f3, d_threadDimensions):
    threadIdx = cuda.grid(1)
    i, j, a, b, k = indices5(threadIdx, d_threadDimensions)
    if i < d_threadDimensions[0]:
        d_res[a, j, i, f3 * b + k] = d_feature[a, j + (b%3) * dilate, i + (b//3) * dilate, k]

@cuda.jit
def GPU_Feature_Bias(d_feature, d_res, dilate, f3, bias, d_threadDimensions):
    threadIdx = cuda.grid(1)
    i, j, a, b, k = indices5(threadIdx, d_threadDimensions)
    if i < d_threadDimensions[0]:
        d_res[a, j, i, f3 * b + k] = d_feature[a, j + (b%3) * dilate, i + (b//3) * dilate, k] + 1 / math.sqrt(f3) * bias

@cuda.jit(device=True)
def indices5(m, threadDimensions):
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
