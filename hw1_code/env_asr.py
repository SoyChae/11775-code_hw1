#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import glob
import nltk

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


# In[ ]:

# In[1]:



from nltk.corpus import stopwords

file_list = 'all_video.lst'

paths = 'asrs/*'
words_list = []
default_stopwords = set(nltk.corpus.stopwords.words('english'))
default_stopwords.add("'ve'")
default_stopwords.add("dr")
default_stopwords.add("d_j")
default_stopwords.add("d_v_ds")
default_stopwords.add("c_ds")
default_stopwords.add("dr")
default_stopwords.add("11")
default_stopwords.add("1950s")
for path in glob.glob(paths):
    if path[-1] != 'm':
        f = open(path, 'r')    
        for line in f.readlines():
            words = line.replace('\n','').replace('.', '').replace(',', '').split(' ') 
            words = [word.replace("'",' ').split()[0] for word in words if len(word) > 1]
            words = [word for word in words if not word[0].isdigit()]
            words = [word.lower() for word in words]
            words = [word for word in words if word not in default_stopwords]
            words_list.extend(words)


words_arr = np.array(words_list)
(vocab, counts) = np.unique(words_arr, return_counts=True)
vocab = vocab[counts.argsort()[::-1][0:500]].tolist()

vocab_size = len(vocab)


# ## ASR feature extraction

# In[6]:


fread = open(file_list, "r")
for line in fread.readlines():
    asr_path = "asrs/" + line.replace('\n','').replace('.mp4','')
    fwrite = open('asrfeat/' + line.replace('\n','').replace('.mp4',''),'w')
    cluster_histogram = np.zeros(vocab_size)
    total_occur = 0

    if os.path.exists(asr_path + '.txt') == True:
        fread = open(asr_path + '.txt', 'r')
        words_list = []
        for L in fread.readlines():
            words = L.replace('\n','').replace('.', '').replace(',', '').split(' ') 
            for i in xrange(len(words)):
                if words[i] in vocab:
                    cluster_histogram[vocab.index(words[i])] += 1
                    total_occur += 1
        fread.close()
    elif os.path.exists(asr_path) == True:
        fread = open(asr_path, 'r')
        words_list = []
        for L in fread.readlines():
            words = L.replace('\n','').replace('.', '').replace(',', '').split(' ') 
            for i in xrange(len(words)):
                if words[i] in vocab:
                    cluster_histogram[vocab.index(words[i])] += 1
                    total_occur += 1
        fread.close()

    if total_occur > 0:
        for m in xrange(vocab_size):
            cluster_histogram[m] /= float(total_occur)
    else:
        cluster_histogram.fill(1.0/vocab_size)

    LL = str(cluster_histogram[0])
    for m in range(1, vocab_size):
        LL += ';' + str(cluster_histogram[m])
    fwrite.write(LL + '\n')
    fwrite.close()

print "ASR features generated successfully!"


# ## SVM train

# In[125]:


pathN = glob.glob('asrfeat/*')
line = pathN[0]
yes_sound_train, no_sound_train = extrYesSoundN('all_trn.lst')
trnLab = extractLabel('all_trn.lst', yes_sound_train)
yes_sound_val, no_sound_val = extrYesSoundN('all_val.lst')
valLab = extractLabel('all_val.lst', yes_sound_val)
X = []
for line in yes_sound_train:
    x = open('asrfeat/{}'.format(line),'r').readline().replace(';',' ').split()
    X.append(np.asarray(x))
lab_ind = []
for i in np.arange(1, 5):
    ind = [j for j, y in enumerate(trnLab) if y == i]
    lab_ind.append(ind)
    
X_sampled = []
tr_lab_sampled = []
for i in range(0, 4):
    if i < 3:
        ind = lab_ind[i]
    else:
        ind = np.random.permutation(len(lab_ind[3]))[0:36]
        
    for j in range(0, len(ind)):
        X_sampled.append(X[ind[j]])
        tr_lab_sampled.append(i+1)
import sklearn
from sklearn.svm import SVC
        
svmModel = SVC(C=1, kernel='rbf', gamma = 26)
svmModel.fit(X_sampled,tr_lab_sampled)


# ## Validation data MAP

# In[126]:


X_val = []
for line in yes_sound_val:
    x = open('asrfeat/{}'.format(line),'r').readline().replace(';',' ').split()
    X_val.append(np.asarray(x))
# Genearte file
fread = open('all_val.lst','r')
vec = 4*np.ones((400,1))
ind = []
## Train data with sound
for line in fread.readlines():
    ind.append(line.split()[0])
    
pred_val = svmModel.predict(X_val)
for i in range(0, len(yes_sound_val)):
    index = ind.index(yes_sound_val[i])
    vec[index] = pred_val[i]
for i in range(1, 5):
    fwrite = open('P00{}_val_pred_asr.txt'.format(i),'w')
    pred_val = np.zeros((400,1))
    indices = [j for j, y in enumerate(vec) if y == i]    
    pred_val[indices] = 1
    for k in range(0, 400):
        fwrite.write(str(int(pred_val[k][0]))+'\n')
    fwrite.close()
    
import os
event = 'P001'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred_asr.txt'.format(event, event)
os.system(command)
event = 'P002'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred_asr.txt'.format(event, event)
os.system(command)
event = 'P003'
command = 'mAP/ap hw1_code/list/{}_val_label {}_val_pred_asr.txt'.format(event, event)
os.system(command)


# ## Test data label generation

# In[134]:


trLabel = 'mfccLabel_test'
fread = open('select_mfcc_higher/{}.csv'.format(trLabel),'r')
x = fread.readlines()
len(x)


# In[127]:


yes_sound_test, no_sound_test = extrYesSoundN('all_test.video')
X_test = []
for line in yes_sound_test:
    x = open('asrfeat/{}'.format(line),'r').readline().replace(';',' ').split()
    X_test.append(np.asarray(x))
# File generation
fread = open('all_test.video','r')
vec = 4*np.ones((1699,1))
ind = []
## File with sound
for line in fread.readlines():
    ind.append(line.split()[0])
pred_test = svmModel.predict(X_test)
for i in range(0, len(yes_sound_test)):
    index = ind.index(yes_sound_test[i])
    vec[index] = pred_test[i]


# In[128]:


FileN = []
FileN.extend(yes_sound_test)
FileN.extend(no_sound_test)


# In[129]:


LAB = ['P001', 'P002', 'P003', 'NULL'] 
fwrite = open('ASR_score.txt'.format(i),'w')
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
