import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import random as rnd

videos=["/truba/home/tukoc/HSVData/Data_For_USC_2008_02_016_glissendo.16.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_028_differents_modes.28.video.avi.mat",
        "/truba/home/tukoc/HSVData/Data_For_USC_2008_02_033_ea.33.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_036_parlee.36.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_046_glissendo.46.video.avi.mat",
        "/truba/home/tukoc/HSVData/Data_For_USC_2008_02_050_ouv_renf.50.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_057_medium.57.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_059_atta_tendue.59.video.avi.mat",
        "/truba/home/tukoc/HSVData/Data_For_USC_2008_02_060_atta_haspire.60.video.avi.mat","/truba/home/tukoc/HSVData/Data_For_USC_2008_02_118_medium.avi.mat"]

# image ve label isimli 2 bos tensor

image=np.empty((27, 0, 300))
label=np.empty((1, 0, 300))

for i in range(len(videos)):
    
    data=scipy.io.loadmat(videos[i])
    
    images=data['ImData']
    labels=data['ImLabel']
    
    image=np.append(image,images,axis=1)
    label=np.append(label,labels,axis=1)
    
    print(i)
    print("image",image.shape)
    print("label",label.shape)
   # print('Glottal area',sum(sum(label)))

image = np.asarray(image).astype('float32')
image=image.reshape(27,21675000)
image=image.transpose()


label = np.asarray(label).astype('float32')
label=label.reshape(1,21675000)
label=label.transpose()

print("image",image.shape)
print("label",label.shape)

IndexData=scipy.io.loadmat('/truba/home/tukoc/HSVData/Index.mat')
Train_Index0=IndexData['Train_Index0']
Train_Index1=IndexData['Train_Index1']
Val_Index0=IndexData['Val_Index0']
Val_Index1=IndexData['Val_Index1']
Test_Index0=IndexData['Test_Index0']
Test_Index1=IndexData['Test_Index1']

print(Train_Index0.shape)
print(Train_Index1.shape)
print(Val_Index0.shape)
print(Val_Index1.shape)
print(Test_Index0.shape)
print(Test_Index1.shape)

#Train_Index=np.concatenate((Train_Index1[0,:],Train_Index0[0,:]))
#Val_Index=np.concatenate((Val_Index1,Val_Index0))
#Test_Index=np.concatenate((Test_Index1,Test_Index0))
Train_Index=np.append(Train_Index1,Train_Index0,axis=1)
Val_Index=np.append(Val_Index1,Val_Index0,axis=1)
Test_Index=np.append(Test_Index1,Test_Index0,axis=1)

print(Train_Index.shape)
print(Val_Index.shape)
print(Test_Index.shape)

print('Training size : ',len(Train_Index),'Validation sample size : ',len(Val_Index),'Test sample size :',len(Test_Index))

#training,validation ve test verilerini ayir.
#N/2 training - N/4 validation - N/4 test
training=image[Train_Index,:]
validate=image[Val_Index,:]
test=image[Test_Index,:]
test1=image[Test_Index1,:]

training_label=label[Train_Index,:]
validate_label=label[Val_Index,:]
test_label=label[Test_Index,:]
test_label1=label[Test_Index1,:]

training=training.reshape(140000,27)
training_label=training_label.reshape(140000,1)
validate=validate.reshape(70000,27)
validate_label=validate_label.reshape(70000,1)
test=test.reshape(21465000,27)
test_label=test_label.reshape(21465000,1)

print("training.shape=       ",training.shape)
print("training_label.shape= ",training_label.shape)
print("validate.shape=       ",validate.shape)
print("validate_label.shape= ",validate_label.shape)
print("test.shape=           ",test.shape)
print("test_label.shape=     ",test_label.shape)
print("test1.shape=           ",test1.shape)
print("test_label1.shape=     ",test_label1.shape)

#scipy.io.savemat('/truba/home/tukoc/HSVData/HSV_Features_27N.mat',{'training':training,'validate':validate,'test':test,'training_label':training_label,'validate_label':validate_label,'test_label':test_label})

# GMM Training

from sklearn import mixture

train0=image[Train_Index0,:]
train1=image[Train_Index1,:]
train0=train0.reshape(70000,27)
train1=train1.reshape(70000,27)

model0 = mixture.GaussianMixture(n_components=128, covariance_type='spherical')
model1 = mixture.GaussianMixture(n_components=128, covariance_type='spherical')

#gmm0 = GMM(n_components=1024,covariance_type="spherical").fit(image[Train_Index0,:])
#gmm1 = GMM(n_components=1024,covariance_type="spherical").fit(image[Train_Index1,:])

model0.fit(train0)
model1.fit(train1)

#labels = gmm.predict(X)

# Validation performans
EstP0=model0.score_samples(validate)
EstP1=model1.score_samples(validate)
LLR=EstP0-EstP1

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

target_name = ['Background', 'Glottis']
report = classification_report(validate_label, LLR<0,target_names=target_name)
print('Validation Set Performance \n\n')
print(report)

# Test performans

EstP0=model0.score_samples(test[:1000000,:])
EstP1=model1.score_samples(test[:1000000,:])
LLR=EstP0-EstP1
print('Test Set Performance \n\n')
report = classification_report(test_label[:1000000,:], LLR<0,target_names=target_name)
print(report)

#scipy.io.savemat('/truba/home/tukoc/HSVData/GMMs_Full_1024.mat',{'model0':model0,'model1':model1})
