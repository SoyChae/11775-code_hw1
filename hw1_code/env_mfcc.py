#!/usr/bin/env python
# coding: utf-8

# ## Default function

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def mfcc_extract(label_name):
    ###################################################################
    exe_opensmile = '/home/ubuntu/tools/opensmile-2.3.0/inst/bin/SMILExtract'
    path_config = '/home/ubuntu/hw1_schae/config/MFCC12_0_D_A.conf'
    trnData = open('{}'.format(label_name))
    trnV = []
    trnL = []
    for line in trnData.readlines():
        tmp = line.split()
        trnV.append(tmp[0])
        trnL.append(tmp[0])

    # Extract features for train, sound 없는 경우 -> zeros vector
    for fn in trnV:
        if os.path.isfile('mfcc_higher/{}.mfcc.csv'.format(fn)) == False:
            # command_A = 'ffmpeg -y -i videos/{}.mp4 -ac 1 -f wav audio/{}.wav'.format(fn, fn)
            # output = os.system(command_A)    
            # if output == 0:            
            if os.path.isfile('audio/{}.wav'.format(fn)) == True:
                command_M = '{} -C {} -I audio/{}.wav -O mfcc_higher/{}.mfcc.csv'.format(exe_opensmile, path_config, fn, fn)
                os.system(command_M)
                print(fn)
        else:
            print('mfcc file already exists')
###################################################################
def extrYesSoundN(fileN):
    fread = open(fileN, 'r')
    yes_sound = []
    no_sound = []
    for line in fread.readlines():
        fn = line.split()[0]
        if os.path.isfile('audio/{}.wav'.format(fn)):
            yes_sound.append(fn)
        else:
            no_sound.append(fn)
    fread.close()
    return yes_sound, no_sound
###################################################################
def mfcc_sampling(dataType, yes_sound):
    lab = []
    ratio = 0.2
    fwrite = open('select_mfcc_higher/select.mfcc_{}.csv'.format(dataType),'w')
    fwrite_trmfcc = open('select_mfcc_higher/mfccLabel_{}.csv'.format(dataType),'w')
    for i in range(0, len(yes_sound)):
        file_list = yes_sound[i]
        fread = open('mfcc_higher/{}'.format(file_list)+'.mfcc.csv','r')
        mfcc_path = "mfcc_higher/" + file_list.replace('\n','') + ".mfcc.csv" 
        array = np.genfromtxt(mfcc_path, delimiter=";")
        frame_dim = array.shape[0]
        feat_dim = array.shape[1]
        extrFeat = range(1,12) + range(14, 25) + range(27, 39)
        NSampledFrame = int(np.floor(frame_dim*ratio))
        sampled_feat = []
        sampled_feat = [array[int(np.floor(frame/ratio))][extrFeat] for frame in range(0, NSampledFrame)]
        
        for j in range(0, int(np.floor(frame_dim*ratio))):
            x = sampled_feat[j][0]
            L = str(x)
            for m in range(1, len(extrFeat)):
                L += ';' + str(sampled_feat[j][m])
            fwrite.write(L + '\n')
            fwrite_trmfcc.write(file_list + '\n')
        print(i)
    fwrite.close()
    fwrite_trmfcc.close()
    print('MFCC_{}_Sampling END'.format(dataType))
###################################################################
def extractLabel(fileN, yesSound):
    fread = open(fileN,'r')
    trLab = []
    for line in fread.readlines():
        L = line.replace('\n','')
        tmp = L.split()
        tmp = np.intersect1d(yesSound, tmp[0])
        if len(tmp) != 0:
            if L[-1] == 'L':
                trLab.append(4)
            else:
                trLab.append(int(L[-1]))

    return trLab
###################################################################
def extractBOW(sel_mfcc, trLabel,loaded_model, Ncluster):
    ## Feature가 축소된 mfcc call
    fread = open('select_mfcc_higher/{}.csv'.format(sel_mfcc),'r')
    X = np.asarray(fread.read().replace(';',' ').split())
    f_lab = open('select_mfcc_higher/{}.csv'.format(trLabel))
    feature_lab = f_lab.read().replace('\n',' ').split()
    video_lab = np.unique(feature_lab)
    Nvideo = len(np.unique(feature_lab))
    ## Data 나열
    fread = open('select_mfcc_higher/{}.csv'.format(sel_mfcc),'r')
    x = []
    for line in fread:
        tmp = line.replace(';',' ').split()
        tmp_vec = []
        for ele in tmp:
            tmp_vec.append(float(ele))
        x.append(tmp_vec)
    centroid_tr = loaded_model.predict(x)
    BOW = []
    for i in range(0, Nvideo):
        dummyVar = video_lab[i]
        indices = [j for j, y in enumerate(feature_lab) if y == dummyVar]
        dummyVar2, __  = np.histogram(centroid_tr[indices], np.arange(0,Ncluster+1))
        N = len(indices)
        pd = [float(dummyVar2[k])/N for k in range(0, Ncluster)]
        BOW.append(pd)
    return BOW


# In[2]:


import os
yes_sound_tr, no_sound_tr = extrYesSoundN('all_trn.lst')
yes_sound_val, no_sound_val = extrYesSoundN('all_val.lst')
yes_sound_test, no_sound_test = extrYesSoundN('all_test.video')
trLab = extractLabel('all_trn.lst', yes_sound_tr)
valLab = extractLabel('all_val.lst', yes_sound_val)


# In[3]:


#mfcc_extract('all_trn.lst')
#mfcc_extract('all_val.lst')
#mfcc_extract('all_test.video')
#mfcc_sampling('trn', yes_sound_tr)
#mfcc_sampling('val', yes_sound_val)
#mfcc_sampling('test', yes_sound_test)


# In[4]:


#kmeans = KMeans(n_clusters=50, init='k-means++', max_iter=300, n_init=10, random_state=0)
#X = np.asarray(fread.read().replace(';',' ').split())
#x = []
#for line in fread:
#    tmp = line.replace(';',' ').replace('\n', '').split()
#    tmp_vec = []
#    for ele in tmp:
#        tmp_vec.append(float(ele))
#    x.append(tmp_vec)
fread = open('select_mfcc_higher/select.mfcc_trn.csv','r')


# In[5]:


filename = 'finalized_model_.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[7]:


BOW_tr = extractBOW('select.mfcc_trn', 'mfccLabel_trn',loaded_model, 50)
BOW_val = extractBOW('select.mfcc_val', 'mfccLabel_val',loaded_model, 50)


# In[8]:


#svc_param_selection(np.asarray(tmp1), np.asarray(tr_lab_sampled), 3)


# In[9]:


#from xgboost import XGBClassifier
#df = pd.DataFrame(data=tmp1)
#df_Val = pd.DataFrame(data=tmp2)
#xgb_classifier = XGBClassifier(n_estimators = 400, learning_rate = 0.001, max_depth = 10)
#xgb_classifier.fit(df, tr_lab_sampled)
#xgb_pred = xgb_classifier.predict(df_Val)
#print(sklearn.metrics.f1_score(valLab, xgb_pred,average='macro'))


# In[10]:


import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample
from sklearn.decomposition import PCA

lab_ind = []
for i in np.arange(1, 5):
    ind = [j for j, y in enumerate(trLab) if y == i]
    lab_ind.append(ind)
    
bow_sampled = []
tr_lab_sampled = []
N_up = 120
for i in range(0, 4):
    if i < 3:
        minority_upsample =[]
        psample = []
        ind = lab_ind[i]
        minor = []
        for j in range(0, len(ind)):
            minor.append(BOW_tr[ind[j]])
        minority_upsample = resample(minor, 
                         replace=True,     # sample with replacement
                         n_samples=N_up,    # to match majority class
                         random_state = 1) # reproducible results
        bow_sampled.extend(minority_upsample)
        tr_lab_sampled.extend(int(i+1)*np.ones(N_up))
        print(np.size(bow_sampled))
    elif i == 3:
        ind = np.random.permutation(len(lab_ind[i]))[0:N_up]
        tmp = []
        for j in range(0, len(ind)):
            tmp.append(BOW_tr[ind[j]])
            tr_lab_sampled.append(i+1)
        bow_sampled.extend(tmp)
        print(np.size(bow_sampled))
tmp1 = bow_sampled
tmp2 = BOW_val

import sklearn
from sklearn.svm import SVC

svmModel = SVC(kernel = 'rbf',C=1, gamma=110)
svmModel.fit(tmp1, tr_lab_sampled)

print(svmModel.score(tmp1, tr_lab_sampled))
print(svmModel.score(tmp2, valLab))
print(sklearn.metrics.f1_score(valLab, svmModel.predict(tmp2),average='macro'))
svmModel.predict(tmp2)[0]
#ind = svmModel.predict(tmp2) == valLab
#idx = [valLab[ind[i]] for i in range(0, len(ind))]
#np.histogram(valLab)
#ind


# In[12]:


filename = 'svm_for_mfcc.sav'
pickle.dump(svmModel, open(filename, 'wb'))


# In[10]:


fread = open('select_mfcc_higher/{}.csv'.format('select.mfcc_val'),'r')
X = np.asarray(fread.read().replace(';',' ').split())
f_lab = open('select_mfcc_higher/{}.csv'.format('mfccLabel_val'))
feature_lab = f_lab.read().replace('\n',' ').split()
video_lab = np.unique(feature_lab)
Nvideo = len(np.unique(feature_lab))

# 파일 생성하자!
fread = open('all_val.lst','r')
vec = np.ones((400,1))
ind = []
## sound가 있는거
for line in fread.readlines():
    ind.append(line.split()[0])
pred_val = svmModel.predict(tmp2)
#pred_val = xgb_classifier.predict(df_Val)
for i in range(0, len(yes_sound_val)):
    index = ind.index(yes_sound_val[i])
    vec[index] = pred_val[i]

pred_val = np.zeros((400,1))
for i in range(1, 5):
    fwrite = open('P00{}_val_pred.txt'.format(i),'w')
    indices = [j for j, y in enumerate(vec) if y == i]    
    pred_val[indices] = 1
    for k in range(0, 400):
        fwrite.write(str(int(pred_val[k][0]))+'\n')
    fwrite.close()

event = 'P001'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred.txt'.format(event, event)
os.system(command)
event = 'P002'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred.txt'.format(event, event)
os.system(command)
event = 'P003'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred.txt'.format(event, event)
os.system(command)


# ## Test data 

# In[6]:


filename = 'svm_for_mfcc.sav'
svmModel = pickle.load(open(filename, 'rb'))


# In[ ]:


sel_mfcc = 'select.mfcc_test'
fread = open('select_mfcc_higher/{}.csv'.format(sel_mfcc),'r')
trLabel = 'mfccLabel_test'
NCluster = 50
fopen = open('centroid_mfcc.csv','w')
cnt= 0
centroid = []

for line in fread.readlines():
    x = line.replace(';',' ').split()
    y = [x[i] for i in range(0, 34)]
    y = np.reshape(y, (1, 34))
    centroid = loaded_model.predict(y)
    fopen.write(str(centroid[0]) + '\n')
    cnt += 1
    print(cnt)
fopen.close()
fread.close()


# sel_mfcc = 'select.mfcc_test'
# fread = open('select_mfcc_higher/{}.csv'.format(sel_mfcc),'r')
# trLabel = 'mfccLabel_test'
# NCluster = 50
# fopen = open('centroid_mfcc.csv','w')
# cnt= 0
# centroid = []
# 
# for line in fread.readlines():
#     x = line.replace(';',' ').split()
#     y = [x[i] for i in range(0, 34)]
#     y = np.reshape(y, (1, 34))
#     centroid = loaded_model.predict(y)
#     fopen.write(str(centroid[0]) + '\n')
#     cnt += 1
#     print(cnt)
# fopen.close()
# fread.close()

# In[10]:


f_lab = open('select_mfcc_higher/{}.csv'.format('mfccLabel_test'))
feature_lab = f_lab.read().replace('\n',' ').split()
video_lab = np.unique(feature_lab)
Nvideo = len(np.unique(feature_lab))


# In[11]:


f_lab = open('select_mfcc_higher/{}.csv'.format('mfccLabel_test'))
feature_lab = f_lab.read().replace('\n',' ').split()
video_lab = np.unique(feature_lab)
Nvideo = len(np.unique(feature_lab))
centroid_ = []
centroid_ = [int(centroid[i]) for i in range(0, len(centroid))]
centroid_ = np.asarray(centroid_)
#extractBOW(sel_mfcc, trLabel,loaded_model, Ncluster
Ncluster = 50
BOW_test = []
for i in range(0, Nvideo):
    dummyVar = video_lab[i]
    indices = [j for j, y in enumerate(feature_lab) if y == dummyVar]
    dummyVar2, __  = np.histogram(centroid_[indices], np.arange(0,Ncluster+1))
    N = len(indices)
    pd = [float(dummyVar2[k])/N for k in range(0, Ncluster)]
    BOW_test.append(pd)
    print(i)


# In[ ]:


# File generation
fread = open('all_test.video','r')
vec = 4*np.ones((1699,1))
ind = []
## File with sound
for line in fread.readlines():
    ind.append(line.split()[0])


# In[ ]:


pred_test = svmModel.predict(BOW_test)
for i in range(0, len(yes_sound_test)):
    index = ind.index(yes_sound_test[i])
    vec[index] = pred_test[i]


# In[ ]:


FileN = []
FileN.extend(yes_sound_test)
FileN.extend(no_sound_test)


# In[ ]:


LAB = ['P001', 'P002', 'P003', 'NULL'] 
fwrite = open('MFCC_score.txt'.format(i),'w')
pred_test = np.zeros((1699,1))
for i in range(1, 5):
    indices = [j for j, y in enumerate(vec) if y == i]
    pred_test[indices] = i
    
for k in range(0, 1699):
        lab = LAB[int(pred_test[k]-1)]
        if int(pred_test[k]) == 4:
            L = 'File {}, {} # classified as {}\n'.format(FileN[k], 0, 'NULL')
        else:
            L = 'File {}, {} # classified as {}\n'.format(FileN[k], int(pred_test[k]), lab.replace('\n',''))
        fwrite.write(L)
fwrite.close()
