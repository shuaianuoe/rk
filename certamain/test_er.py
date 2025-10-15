# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import srk
import pickle
import os
import time
from utils import  compute_con_acc, compute_faithfulness_er, compute_faithfulness_er_fnum, alg_config_parse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='spacy')

def save_model(model, modeldir):

    os.makedirs(os.path.dirname(modeldir), exist_ok=True)
    with open(modeldir, 'wb') as f:
        pickle.dump(model, f)

def load_model(modeldir):

    if os.path.exists(modeldir):
        print("Model file is found!")
        with open(modeldir, 'rb') as f:
            return pickle.load(f)
    else:
        print("Model file not found!")
        return None

def predict_fn(x, **kwargs):
    return model.predict(x, **kwargs)


#%%       
alg_dict = alg_config_parse('../config.yaml')            

dataset = alg_dict['datasetsname']
sample_num = alg_dict['sample_num']

test_df = pd.read_csv('../data_process/'+dataset+'_test.csv')
sample_num = min(sample_num, test_df.shape[0])

modeldir = '../certamain/model/ditto/' + dataset + '/model.pkl'  
model = load_model(modeldir)

test_df = test_df.drop(['label','ltable_id','rtable_id'], axis=1)
columns_name_list = test_df.columns.values.tolist() 
X = columns_name_list[0:-1]
Y = columns_name_list[-1]
k = 5


res_dict = {}
s_time = np.zeros(sample_num, dtype='float')
exp_s = np.zeros(sample_num, dtype='int')
precision_s = np.zeros(sample_num, dtype='float')
conformity_s = np.zeros(sample_num, dtype='bool')
faithfulness_s = np.zeros(sample_num, dtype='float')
fidelity_s = np.zeros(sample_num, dtype='float')


for instance_id in range(test_df.shape[0]):
    start_time = time.time()
    if instance_id % 20 == 0:  
        print("instance_id", instance_id)
    if instance_id >= sample_num:
        break
    instance_value = test_df.loc[instance_id]

    subsets = {}
    for col in X:
        diff_indices = test_df.index[test_df[col] != instance_value[col]].tolist()
        subsets[col] = set(diff_indices)

    universe = set(test_df.index[test_df[Y] != instance_value[Y]].tolist())
    cover_sets = srk.greedy_set_cover(universe, subsets, 0)
        
    res_dict[instance_id] = cover_sets
    exp_s[instance_id] = len(cover_sets)  
    s_time[instance_id] = time.time() - start_time
 
    precision, conformity = compute_con_acc(test_df, instance_value, cover_sets)
    precision_s[instance_id] = precision
    conformity_s[instance_id] = conformity
    
    faithfulness = compute_faithfulness_er_fnum(test_df, instance_value, cover_sets, predict_fn)
    faithfulness_s[instance_id] = faithfulness
    
    with open('exp/'+dataset+'_srk-expsize.pkl', 'wb') as f:
       pickle.dump(exp_s, f)

    
        
print("mean_time:", round(np.mean(s_time), 3)) 

print("min_size:", np.min(exp_s))
print("max_size:", np.max(exp_s))
print("mean_size:", round(np.mean(exp_s),3))

print("mean_conformity:", round(np.mean(conformity_s), 3)) 
print("mean_precision:", round(np.mean(precision_s), 3)) 
print("mean_faithfulness:", round(np.mean(faithfulness_s), 3)) 
