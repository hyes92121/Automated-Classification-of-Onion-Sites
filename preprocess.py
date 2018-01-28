
# coding: utf-8

# In[1]:


import os
import sys
import subprocess

PARAMS_DIR = 'parameters/'
WORD_GRP   = sys.argv[1]
dirs  = [x for x in os.listdir('data') if 'tr_' in x]


# In[2]:


# generate title
with open(PARAMS_DIR+'title.txt', 'w', encoding='utf-8') as f:    
    for cat in dirs:
        cat_path = 'data/'+cat
        for d in [f for f in os.listdir(cat_path) if '.DS' not in f]:
            with open('{}/{},,,,,,,,,,'.format(cat_path, d), 'r', encoding='utf-8') as ff:
                title = ff.readline().rstrip()
            f.write('{},,{}\n'.format(d, title))


# In[3]:


#generate training labeled data
with open(PARAMS_DIR+'train.txt', 'w', encoding='utf-8') as f:    
    for cat in dirs:
        cat_path = 'data/'+cat
        for d in [f for f in os.listdir(cat_path) if '.DS' not in f]:
            f.write('{},{}\n'.format(d, cat[3:]))


# In[4]:


#generate testing data
with open(PARAMS_DIR+'test.txt', 'w', encoding='utf-8') as f:    
    for cat in ['{}{}'.format('te_', d[3:]) for d in dirs]:
        cat_path = 'data/'+cat
        for d in [f for f in os.listdir(cat_path) if '.DS' not in f]:
            f.write('{},{}\n'.format(d, cat[3:]))


# In[5]:


# generate word groups
for cat in dirs:
    cat_path = 'data/'+cat
    for d in [f for f in os.listdir(cat_path) if '.DS' not in f]:
        process = subprocess.Popen(['java', 'GenerateWordGrp', '{}/{}'.format(cat_path, d)], stdout=subprocess.PIPE)
        stdout = process.communicate()[0]
        wrdgrp = stdout.decode('UTF-8').split('\n')
        with open('{}/{}.{}'.format('wrdgroups', d, 'onion'), 'w', encoding='utf-8') as ff:
            for wrd in wrdgrp:
                ff.write('{}\n'.format(wrd))            
            

