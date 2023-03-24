# Saab transformation for PixelHop unit
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA
import pickle
import time

class Saab():
    def __init__(self, pca_name, kernel_sizes,  energy_percent=None, useDC=False):
        self.pca_name = pca_name
        self.kernel_sizes = kernel_sizes
        self.useDC = useDC
        self.energy_percent = energy_percent

    def remove_mean(self, features, axis):
        feature_mean = np.mean(features, axis=axis, keepdims=True)
        feature_remove_mean = features - feature_mean
        return feature_remove_mean, feature_mean
    
    def find_kernels_pca(self, samples,  energy_percent, N=10000000):
        # control the number of patches, eliminate low variance patches
        N=min(N,samples.shape[0])
        samples=samples[np.random.choice(samples.shape[0],N,replace=False)]
        var_samples=np.var(samples,1)
        samples=samples[var_samples>1e-4]

        pca = PCA(n_components=samples.shape[1], svd_solver='full')
        pca.fit(samples)
        
        if energy_percent:
            energy = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.sum(energy < energy_percent) + 1
        kernels = pca.components_[:num_components, :]
    
        mean = pca.mean_
        print("       <Info>        Num of kernels: %d" % num_components)
        print("       <Info>        Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components - 1])
        return kernels, mean

    def Saab_transform(self, pixelhop_feature, kernel_sizes, energy_percent, useDC): 
        S = pixelhop_feature.shape
        print("       <Info>        pixelhop_feature.shape: %s"%str(pixelhop_feature.shape))

        sample_patches = pixelhop_feature.reshape(S[0]*S[1]*S[2],-1)
        
        pca_params = {}
        pca_params['kernel_size'] = kernel_sizes

        sample_patches_centered, feature_expectation = self.remove_mean(sample_patches, axis=0)
        training_data, dc = self.remove_mean(sample_patches_centered, axis=1)
        print('       <Info>        training_data.shape: {}'.format(training_data.shape))
        
        kernels, mean = self.find_kernels_pca(training_data, energy_percent)

        transformed = np.matmul(sample_patches_centered, np.transpose(kernels))
        bias = LA.norm(transformed, axis=1)
        bias = np.max(bias)
        
        pca_params['Layer_%d/bias' % 0] = bias

        print("       <Info>        Sample patches shape after flatten: %s"%str(sample_patches.shape))
        print("       <Info>        Kernel shape: %s"%str(kernels.shape))
        print("       <Info>        Transformed shape: %s"%str(transformed.shape))
        pca_params['Layer_%d/feature_expectation' % 0] = feature_expectation
        pca_params['Layer_%d/kernel' % 0] = kernels
        pca_params['Layer_%d/pca_mean' % 0] = mean
        return pca_params

    def fit(self, pixelhop_feature):
        print("------------------- Start: Saab transformation")
        t0 = time.time()
        pca_params = self.Saab_transform(pixelhop_feature=pixelhop_feature,
                                                kernel_sizes=self.kernel_sizes,
                                                energy_percent=self.energy_percent, 
                                                useDC=self.useDC)
        fw = open(self.pca_name, 'wb')
        pickle.dump(pca_params, fw)
        fw.close()
        print("       <Info>        Save pca params as name: %s"%str(self.pca_name))
        print("------------------- End: Saab transformation -> using %10f seconds"%(time.time()-t0))    


