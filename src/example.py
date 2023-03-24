"""
@author: Vinay Kadam
Training and testing for semantic segmentation of mouse heart using PixelHop 
Dataset info: Light sheet fluroscence microscopy (LSFM) dataset from D-incubator https://labs.utdallas.edu/d-incubator/
This code uses 1024x1024 images/masks. Patches of 64x64 from images and labels have been extracted and fed to the PixelHop algorithm.  

"""

import numpy as np 
import os, glob
from framework.layer import *
from framework.pixelhop import *
from utils import loadImage
import imageio
from division import div
import timeit
import xgboost as xgb
from sklearn import preprocessing
import pickle
from sklearn.feature_selection import f_classif, SelectPercentile
from datetime import timedelta

SAVE={}
img_size = 1024
patch_size = 64#16
delta_x = patch_size # non overlap

num_training_imgs = 1
train_img_path = r'C:/Users/Austin/Desktop/PixelHop/data/train/*.png'
test_img_path =  r'C:/Users/Austin/Desktop/PixelHop/data/test/*.png' 

train_img_addrs = glob.glob(train_img_path)
test_img_addrs = glob.glob(test_img_path)

print(train_img_path)
print(train_img_addrs)




def run():
    
    start_train = timeit.default_timer()

    print("----------------------TRAINING-------------------------")
    # Initialize
    mask_patch_list, img_patch_list = [], []

    # Control the num of training images
    count = 0
    for train_img_addr in train_img_addrs:
        if 'raw' in train_img_addr: # only want to load training images that are raw
            count += 1
            if count > num_training_imgs: break
            print('Adding {} for training................'.format(train_img_addr))

            # Add raw images
            img = loadImage(train_img_addr) # Load image
            print(train_img_addr)

            # Add mask images
            mask = loadImage(train_img_addr.replace('raw', 'seg'))
            print(mask.shape)

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)
                print(img.shape)

            # Create patches for training images
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        img_patch = img[i:i+patch_size, j:j+patch_size, :]
                        mask_patch = mask[i:i+patch_size, j:j+patch_size]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:j+patch_size], ((0,patch_size-temp_size[0]),(0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:i+patch_size, j:], ((0,0),(0,patch_size-temp_size[1])), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                        mask_patch = np.lib.pad(mask[i:, j:], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1])), 'edge')

                    assert (img_patch.shape[0], img_patch.shape[1]) == (patch_size,patch_size)

                    # Save each patch to list
                    img_patch_list.append(img_patch)
                    mask_patch_list.append(mask_patch)
                    
                    
    # Convert list to numpy array
    img_patches = np.asarray(img_patch_list)
    mask_patches = np.array(mask_patch_list)
    print(img_patches.shape)
    print(mask_patches.shape)

    print('--------------------------------------')
    # Number of classes
    print('NUmber of classes: {}'.format(np.unique(mask_patches)))
    

    ################################################## PIXELHOP UNIT 1 ####################################################
        
    train_feature1=PixelHop_Unit(img_patches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=1, energypercent=0.98)
    
    ################################################ PIXELHOP UNIT 2 ####################################################

    train_featurem1 = MaxPooling(train_feature1)
    train_feature2=PixelHop_Unit(train_featurem1, dilate=1, pad='reflect',  weight_name='pixelhop2.pkl', getK=1, energypercent=0.98)
   
    
    print(train_feature1.shape)
    print(train_feature2.shape)
    
    # Upsample the pixelhop feature 
    train_feature_reduce_unit1 = train_feature1 
    train_feature_reduce_unit2 = myResize(train_feature2, img_patches.shape[1], img_patches.shape[2])
    print(train_feature_reduce_unit1.shape)
    print(train_feature_reduce_unit2.shape)

    # Reshape the pixelhop feature
    train_feature_unit1 = train_feature_reduce_unit1.reshape(train_feature_reduce_unit1.shape[0]*train_feature_reduce_unit1.shape[1]*train_feature_reduce_unit1.shape[2], -1)
    train_feature_unit2 = train_feature_reduce_unit2.reshape(train_feature_reduce_unit2.shape[0]*train_feature_reduce_unit2.shape[1]*train_feature_reduce_unit2.shape[2], -1)
   
    del train_feature_reduce_unit1, train_feature_reduce_unit2
    
    print(train_feature_unit1.shape)
    print(train_feature_unit2.shape)
    
    # Aligned all the features along with the ground truth
    feature_list_unit1, feature_list_unit2, gt_list = [], [], []
    patch_ind = 0
    # Control the num of training images
    count = 0
    for train_img_addr in train_img_addrs:
        count += 1
        if count > num_training_imgs: break
        for i in range(0, img_size, delta_x):
            for j in range(0, img_size, delta_x):
                # for each patch
                for k in range(patch_size):
                    for l in range(patch_size):
                        gt = mask_patch_list[patch_ind][k,l]
                        # get features
                        feature1 = np.array([])
                        # subpatch_ind = patch_ind*(div(patch_size,subpatch_size))*(div(patch_size,subpatch_size)) + (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + l//subpatch_size 
                        feature1 = np.append(feature1, img_patch_list[patch_ind][k,l,:]) # intensity feature
                        feature1 = np.append(feature1, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) # positional feature
                        
                        feature2 = np.array([])
                        feature2 = np.append(feature2, img_patch_list[patch_ind][k,l,:]) # intensity feature
                        feature2 = np.append(feature2, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) # positional feature
                        

                        # Add all the features together
                        feature1 = np.append(feature1, train_feature_unit1[patch_ind,:])   
                        feature2 = np.append(feature2, train_feature_unit2[patch_ind,:])   

                        feature_list_unit1.append(feature1)
                        feature_list_unit2.append(feature2)

                        gt_list.append(gt)
                patch_ind += 1

    del feature1, feature2
    
    gt_list = np.array(gt_list)
    feature_list_unit1 = np.array(feature_list_unit1) 
    feature_list_unit2 = np.array(feature_list_unit2)  
    
    print(feature_list_unit1.shape)
    print(feature_list_unit2.shape)
    
    print(gt_list.shape)
    
    # F-test to get top 80% features
    fs1 = SelectPercentile(score_func=f_classif, percentile=80)
    fs1.fit(feature_list_unit1, gt_list)
    new_features1 = fs1.transform(feature_list_unit1)
    print(new_features1.shape)
    print(fs1.scores_)
    
    fs2 = SelectPercentile(score_func=f_classif, percentile=80)
    fs2.fit(feature_list_unit2, gt_list)
    new_features2 = fs2.transform(feature_list_unit2)
    print(new_features2.shape)
    print(fs2.scores_)
    
    # Concatenate all the features together
    concat_features  = np.concatenate((new_features1, new_features2), axis=1)
    print(concat_features.shape)
    
    del feature_list_unit1, new_features1, feature_list_unit2, new_features2
    
    # Preprocessing (standardize features by removing the mean and scaling to unit variance)
    scaler1=preprocessing.StandardScaler().fit(concat_features)
    feature = scaler1.transform(concat_features) 
    print(feature.shape)

    # Define and train XGBoost algorithm
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", verbosity=3)
    clf = xgb_model.fit(feature, gt_list)

    # Save all the model files
    pickle.dump(clf, open("C:/Users/Austin/Desktop/PixelHop/results/classifier.sav",'wb'))
    pickle.dump(scaler1, open("C:/Users/Austin/Desktop/PixelHop/results/scaler1.sav",'wb'))
    pickle.dump(fs1, open("C:/Users/Austin/Desktop/PixelHop/results/fs1.sav",'wb'))
    pickle.dump(fs2, open("C:/Users/Austin/Desktop/PixelHop/results/fs2.sav",'wb'))
    print('All models saved!!!')

    stop_train = timeit.default_timer()

    print('Total Time: ' + str(timedelta(seconds=stop_train-start_train)))
    f = open('C:/Users/Austin/Desktop/PixelHop/results/train_time.txt','w+')
    f.write('Total Time: ' + str(timedelta(seconds=stop_train-start_train))+'\n')
    f.close()



    start_test = timeit.default_timer()
    print("----------------------TESTING-------------------------")

    test_img_patch, test_img_patch_list = [], []
    count = 0
    for test_img_addr in test_img_addrs:
        if 'raw' in test_img_addr: # only want to load testing images that are raw
            count += 1
            print('Processing {}............'.format(test_img_addr))

            img = loadImage(test_img_addr) 

            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2)

            # Initialzing
            predict_0or1 = np.zeros((img_size, img_size, 2))

            predict_mask = np.zeros(img.shape)

            # Create patches for test image
            for i in range(0, img_size, delta_x):
                for j in range(0, img_size, delta_x):
                    if i+patch_size <= img_size and j+patch_size <= img_size:
                        test_img_patch = img[i:i+patch_size, j:j+patch_size, :]

                    elif i+patch_size > img_size and j+patch_size <= img_size:
                        temp_size = img[i:, j:j+patch_size, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:j+patch_size, :], ((0,patch_size-temp_size[0]),(0,0), (0,0)), 'edge')

                    elif i+patch_size <= img_size and j+patch_size > img_size:
                        temp_size = img[i:i+patch_size, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:i+patch_size, j:, :], ((0,0),(0,patch_size-temp_size[1]), (0,0)), 'edge')

                    else: 
                        temp_size = img[i:, j:, :].shape
                        test_img_patch = np.lib.pad(img[i:, j:, :], ((0,patch_size-temp_size[0]),(0,patch_size-temp_size[1]), (0,0)), 'edge')
                    assert (test_img_patch.shape[0], test_img_patch.shape[1]) == (patch_size,patch_size)
                    
                    test_img_patch_list.append(test_img_patch)
                    

                    # convert list to numpy
                    test_img_subpatches = np.asarray(test_img_patch_list)
                    print(test_img_subpatches.shape)

                    
                    ################################################## PIXELHOP UNIT 1 ####################################################
        
                    test_feature1=PixelHop_Unit(test_img_subpatches, dilate=1, pad='reflect', weight_name='pixelhop1.pkl', getK=0, energypercent=0.98)
                   
        
                    
                    ################################################# PIXELHOP UNIT 2 ####################################################
                    test_featurem1 = MaxPooling(test_feature1)
                    test_feature2=PixelHop_Unit(test_featurem1, dilate=1, pad='reflect', weight_name='pixelhop2.pkl', getK=0, energypercent=0.98)
                    
    
                    test_feature_reduce_unit1 = test_feature1 
                    test_feature_reduce_unit2 = myResize(test_feature2, test_img_subpatches.shape[1], test_img_subpatches.shape[2])
                    print(test_feature_reduce_unit1.shape)
                    print(test_feature_reduce_unit2.shape)

                    
                    test_feature_unit1 = test_feature_reduce_unit1.reshape(test_feature_reduce_unit1.shape[0]*test_feature_reduce_unit1.shape[1]*test_feature_reduce_unit1.shape[2], -1)
                    test_feature_unit2 = test_feature_reduce_unit2.reshape(test_feature_reduce_unit2.shape[0]*test_feature_reduce_unit2.shape[1]*test_feature_reduce_unit2.shape[2], -1)
                    
                    print(test_feature_unit1.shape)
                    print(test_feature_unit2.shape)
    
                    
                    #--------lag unit--------------
                    test_feature_list_unit1, test_feature_list_unit2, test_feature_list_unit3, test_feature_list_unit4 = [], [], [], []
                    
                    for k in range(patch_size):
                        for l in range(patch_size):
                            ######################################
                            # get features
                            feature1 = np.array([])
                            # patch_ind = (div(k,patch_size))*(div(patch_size,patch_size)) + div(l,patch_size) # int div
                            # subpatch_ind = (div(k,subpatch_size))*(div(patch_size,subpatch_size)) + div(l,subpatch_size) # int div
                            feature1 = np.append(feature1, test_img_patch[k,l,:])
                            feature1 = np.append(feature1, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
                            feature2 = np.array([])
                            feature2 = np.append(feature2, test_img_patch[k,l,:])
                            feature2 = np.append(feature2, [div(patch_size,2) - abs(i+k-div(patch_size,2)), div(patch_size,2) - abs(j+l-div(patch_size,2))]) #int div
                            
                            
                            # feature = np.append(feature, test_feature12[0,:]) #takes only first few comps
                            feature1 = np.append(feature1, test_feature_unit1[patch_ind,:])
                            feature2 = np.append(feature2, test_feature_unit2[patch_ind,:])

                            test_feature_list_unit1.append(feature1)
                            test_feature_list_unit2.append(feature2)
                            
                            
                    del feature1, feature2#, feature3, feature4
                    
                    
                    feature_list_unit1 = np.array(test_feature_list_unit1)
                    feature_list_unit2 = np.array(test_feature_list_unit2)
                    
                    
                    
                    print(feature_list_unit1.shape)
                    print(feature_list_unit2.shape)


                    test_feature_red1= fs1.transform(feature_list_unit1)
                    test_feature_red2= fs2.transform(feature_list_unit2)


                    test_concat_features  =np.concatenate((test_feature_red1, test_feature_red2), axis=1)
                    print(test_concat_features.shape)
                    

                    feature_test = scaler1.transform(test_concat_features)
                    print(feature_test.shape)
                    

                    pre_list = clf.predict(feature_test)
                    print(pre_list.shape)
                    print(np.unique(pre_list))

                    # Generate predicted result
                    for k in range(patch_size):
                        for l in range(patch_size):
                            if i+k >= img_size or j+l >= img_size: break

                            # Binary
                            if pre_list[k*patch_size + l] > 0.5:
                                predict_0or1[i+k, j+l, 1] += 1
                            else:
                                predict_0or1[i+k, j+l, 0] += 1

                            # Multi-class
                            # if pre_list[k*patch_size + l] == 85.0:
                            #     predict_0or1[i+k, j+l, 1] += 1
                            # if pre_list[k*patch_size + l] == 170.0:
                            #     predict_0or1[i+k, j+l, 2] += 1
                            # if pre_list[k*patch_size + l] == 255.0:
                            #     predict_0or1[i+k, j+l, 3] += 1
                            # else:
                            #     predict_0or1[i+k, j+l, 0] += 1

            print('*************************************************************************************')
            print('one predicted mask')
            predict_mask = np.argmax(predict_0or1, axis=2)


            imageio.imwrite('C:/Users/Austin/Desktop/PixelHop/results/'+os.path.basename(test_img_addr), predict_mask)

    
    stop_test = timeit.default_timer()

    print('Total Time: ' + str(timedelta(seconds=stop_test-start_test)))
    f = open('C:/Users/Austin/Desktop/PixelHop/results/test_time.txt','w+')
    f.write('Total Time: ' + str(timedelta(seconds=stop_test-start_test))+'\n')
    f.close()

    
if __name__=="__main__":

    run()
    