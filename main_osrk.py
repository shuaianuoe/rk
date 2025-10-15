# -*- coding: utf-8 -*-

import numpy as np
import time
import pandas as pd
import osrk as osc
from utils import alg_config_parse, compute_con_acc
import os

alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
epsilon = alg_dict['epsilon']
sample_num = alg_dict['sample_num']
online_fraction = alg_dict['online_fraction']

print("*"*20)
print("dataset:", datasetsname)

data_df = pd.read_csv('data_process/'+datasetsname+'_test.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
data_df = data_df.drop('Target', axis=1)

columns_name_list = data_df.columns.values.tolist()   
X = columns_name_list[0:-1]
Y = columns_name_list[-1]

# Record the explanation of each sample to be explained
res_dict = {}
s_time = np.zeros(sample_num, dtype='float')
exp_s = np.zeros(sample_num, dtype='int')
consistency_s = np.zeros(sample_num, dtype='float')
acc_s = np.zeros(sample_num, dtype='bool')

data_df = data_df.sample(frac=1).reset_index(drop=True)
for beexplain_id in range(data_df.shape[0]):
    # current tuple to be explained
    print("beexplain_id:", beexplain_id)
    if beexplain_id >= sample_num:
        break
    instance_value = data_df.loc[beexplain_id]
    # construct empty subsets
    sample_subsets = {key: set() for key in X}
    # construct empty universe
    universe_set = set()
    # Initializing
    final_subsets, cover_elements, sample_subsets_weight = osc.ini_rand_sc(data_df, X, Y, sample_subsets)
    
    start_time = time.time() * 1000
    alg_time = 0
    online_num = data_df.shape[0]
    for instance_id in range(int((online_num-1)*online_fraction)):
        # print("beexplain_id:", instance_id)
        current_insid = instance_id
        # compute the subsets and universe
        if data_df.loc[beexplain_id, Y] == data_df.loc[current_insid, Y]:
            continue
        else:
            universe_set.add(current_insid)    
        for feature_name in X:
            if data_df.loc[beexplain_id, feature_name] != data_df.loc[current_insid, feature_name]:
                sample_subsets[feature_name].add(current_insid)
                # Key: If the current feature has already been selected before, automatically update the cover_ Elements
                if feature_name in final_subsets:
                    cover_elements.add(current_insid)                  
        alg_start_time = time.time() 
        final_subsets = osc.rand_sc(epsilon, current_insid, universe_set, cover_elements, sample_subsets, sample_subsets_weight, final_subsets)
        alg_time += time.time() * 1000 - alg_start_time
     
    res_dict[beexplain_id] = final_subsets
    exp_s[beexplain_id] = len(final_subsets)  
    s_time[beexplain_id] = alg_time
    
    consistency, acc = compute_con_acc(data_df.iloc[:int(data_df.shape[0]*online_fraction)], instance_value, final_subsets)
    consistency_s[beexplain_id] = consistency
    acc_s[beexplain_id] = acc


print("*"*20)     
print("min_size:", np.min(exp_s))
print("max_size:", np.max(exp_s))
print("mean_size:", round(np.mean(exp_s),2))

print("min_time:", round(np.min(s_time), 2))
print("max_time:", round(np.max(s_time), 2))
print("mean_time:", round(np.mean(s_time), 2)) 

print("mean_precision:", round(np.mean(acc_s), 3)) 

print("mean_conformity:", round(np.mean(consistency_s), 3)) 

print("relative keys:", res_dict)


# store results
dir_path = "results"  
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
key_df = pd.DataFrame.from_dict(res_dict, orient='index')
key_df.columns = ['feature' + str(i+1) for i in range(key_df.shape[1])]
res_df = pd.concat([data_df.loc[0:sample_num-1, :], key_df], axis=1)
res_df.to_csv('results/osrk_'+datasetsname+'.csv')