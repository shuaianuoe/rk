# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd
import osrk as osc
import os
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils import alg_config_parse, check_exp

                     
alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
epsilon = alg_dict['epsilon']
sample_num = alg_dict['sample_num']
online_fraction = alg_dict['online_fraction']
noise_flag = alg_dict['noise_flag']

print("*"*20)
print("dataset:", datasetsname)

data_df = pd.read_csv('data_process/'+datasetsname+'_test.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
y_test = data_df['Target']
data_df = data_df.drop('Target', axis=1)

X_test = data_df.iloc[:, :-1]
y_pred = data_df.iloc[:, -1]

y_test_splits = np.array_split(y_test, 5)
y_pred_splits = np.array_split(y_pred, 5)

for i, (y_test_split, y_pred_split) in enumerate(zip(y_test_splits, y_pred_splits)):
    accuracy = accuracy_score(y_test_split, y_pred_split)
    print(f'Accuracy for split {i+1}: {accuracy*100:.2f}%')
    
#%% Without noise: We observe the proportion of explained size as the proportion of test samples reaches 20%, 40%, and 100%
columns_name_list = data_df.columns.values.tolist()   
X_name = columns_name_list[0:-1]
Y_name = columns_name_list[-1]

res_dict = {}
s_time = np.zeros(sample_num+1, dtype='float')
exp_s = np.zeros(sample_num+1, dtype='int')

for beexplain_id in range(X_test.shape[0]):
    print("beexplain_id:", beexplain_id)
    if beexplain_id > sample_num:
        break
    # construct empty subsets
    sample_subsets = {key: set() for key in X_name}
    # construct empty universe
    universe_set = set()
    final_subsets, cover_elements, sample_subsets_weight = osc.ini_rand_sc(X_test, X_name, Y_name, sample_subsets)
    
    start_time = time.time() * 1000
    alg_time = 0
    online_num = X_test.shape[0]
    for current_insid in range(int((online_num)*online_fraction)):
        if y_pred[beexplain_id] == y_pred[current_insid]:
            continue
        else:
            universe_set.add(current_insid)       
        for feature_name in X_name:
            if X_test.loc[beexplain_id, feature_name] != X_test.loc[current_insid, feature_name]:
                sample_subsets[feature_name].add(current_insid)
                if feature_name in final_subsets:
                    cover_elements.add(current_insid)                  
        alg_start_time = time.time() * 1000
        final_subsets = osc.rand_sc(epsilon, current_insid, universe_set, cover_elements, sample_subsets, sample_subsets_weight, final_subsets)
        alg_time += time.time() * 1000-alg_start_time
     
    res_dict[beexplain_id] = final_subsets
    exp_s[beexplain_id] = len(final_subsets)  
    s_time[beexplain_id] = alg_time
    
    # if epsilon==0 and online_fraction==1:    
    #     check_exp(data_df.loc[0:online_num, :], beexplain_id, final_subsets)    


print("min size:", np.min(exp_s))
print("max size:", np.max(exp_s))
print("mean size:", np.mean(exp_s))
print("min time:", round(np.min(s_time), 3))
print("max time:", round(np.max(s_time), 3))
print("mean time:", round(np.mean(s_time), 3))

            
#%% Add noise: Add noise to the last 40% of the samples.
#The specific approach is to obtain the explanation for a sample at the top 60% and then add noise to the samples between 60% and 100%
#Starting from 0 to obtain 60% explanation: Based on this 60% explanation, we add noise to the remaining 40% of the samples. After adding the noise, we will continue to calculate the size

# Specify the file path to save or load the model
model_file_path = 'results/model/'+datasetsname+'_xgb_model.json'

if noise_flag==True and online_fraction>0.6:
    # Check if the model file exists
    if os.path.exists(model_file_path):
        # Load the model from the file
        print('Loading the existing model...')
        best_model = XGBClassifier()
        best_model.load_model(model_file_path)
        
    print("*"*20)
    print("test nosiy case************************")
    res_noise_dict = {}
    s_noise_time = np.zeros(sample_num+1, dtype='float')
    exp_s_noise = np.zeros(sample_num+1, dtype='int')
    
    for beexplain_id in range(X_test.shape[0]):
        print("beexplain_id:", beexplain_id)
        if beexplain_id > sample_num:
            break
        # construct empty subsets
        sample_subsets = {key: set() for key in X_name}
        # construct empty universe
        universe_set = set()
        final_subsets, cover_elements, sample_subsets_weight = osc.ini_rand_sc(X_test, X_name, Y_name, sample_subsets)
        
        start_time = time.time() 
        alg_time = 0
        online_num = X_test.shape[0]
        # 这里 online_fraction 设置为固定的0.6
        for instance_id in range(int((online_num-1)*0.6)):
            # print("beexplain_id:", instance_id)
            current_insid = instance_id
            if y_pred[beexplain_id] == y_pred[current_insid]:
                continue
            else:
                universe_set.add(current_insid)            
            for feature_name in X_name:
                if X_test.loc[beexplain_id, feature_name] != X_test.loc[current_insid, feature_name]:
                    sample_subsets[feature_name].add(current_insid)
                    if feature_name in final_subsets:
                        cover_elements.add(current_insid)
                        
            alg_start_time = time.time() * 1000
            final_subsets = osc.rand_sc(epsilon, current_insid, universe_set, cover_elements, sample_subsets, sample_subsets_weight, final_subsets)
            alg_time += time.time()* 1000-alg_start_time
        
        #%% Add noise to the last 40% of the data
        X_test_new = X_test.iloc[int(X_test.shape[0]*0.6):]  
        X_beexp = X_test.loc[beexplain_id]
        prob = 0.5
        def modify_row(row):
            for col in X_test_new.columns:
                if col in final_subsets and row[col] != X_beexp[col]:
                    if np.random.rand() < prob:  
                        row[col] = X_beexp[col]
                elif col not in final_subsets and row[col] == X_beexp[col]:
                    if np.random.rand() < prob:  
                        unique_values = X_test_new[col].unique()
                        unique_values = unique_values[unique_values != X_beexp[col]]
                        if len(unique_values) > 0:
                            row[col] = np.random.choice(unique_values)
            return row
        
        X_test_new_copy = X_test_new.copy()
        X_test_new_modified = X_test_new_copy.apply(modify_row, axis=1)
        X_test_com = pd.concat([X_test.iloc[:int(X_test.shape[0]*0.6)] , X_test_new_modified], axis=0)
        y_pred_new = best_model.predict(X_test_com)
        
        #%% Add noise to the last 40% of the data
        for instance_id in range(int((online_num-1)*0.6), online_num-1):
            # print("beexplain_id:", instance_id)
            current_insid = instance_id 
            # compute the subsets and universe
            if y_pred[beexplain_id] == y_pred_new[current_insid]:
                continue
            else:
                universe_set.add(current_insid)               
            for feature_name in X_name:
                if X_test_com.loc[beexplain_id, feature_name] != X_test_com.loc[current_insid, feature_name]:
                    sample_subsets[feature_name].add(current_insid)
                    if feature_name in final_subsets:
                        cover_elements.add(current_insid)
                        
            alg_start_time = time.time() * 1000
            final_subsets = osc.rand_sc(epsilon, current_insid, universe_set, cover_elements, sample_subsets, sample_subsets_weight, final_subsets)
            alg_time += time.time()* 1000-alg_start_time
            
        res_noise_dict[beexplain_id] = final_subsets
        exp_s_noise[beexplain_id] = len(final_subsets)  
        s_noise_time[beexplain_id] = alg_time
     
        
    print("min size:", np.min(exp_s_noise))
    print("max size:", np.max(exp_s_noise))
    print("mean size:", np.mean(exp_s_noise))
    print("min time:", round(np.min(s_noise_time), 3))
    print("max time:", round(np.max(s_noise_time), 3))
    print("mean time:", round(np.mean(s_noise_time), 3))
    
    X_test_splits = np.array_split(X_test_com, 5)
    y_test_splits = np.array_split(y_test, 5)
    
    for i, (X_test_split, y_test_split) in enumerate(zip(X_test_splits, y_test_splits)):
        y_pred = best_model.predict(X_test_split)
        accuracy = accuracy_score(y_test_split, y_pred)
        print(f'Accuracy for split {i+1}: {accuracy*100:.2f}%')













