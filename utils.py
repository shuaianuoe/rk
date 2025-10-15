# -*- coding: utf-8 -*-

import yaml
import csv
import os
import random
import numpy as np

def check_exp(data_df, beexplain_id, final_subsets):
    beexplain_value = data_df.loc[beexplain_id]
    df = data_df.copy()
    
    for key in final_subsets:
        #  print(key)
        df = df.loc[(df[key] == beexplain_value[key])]
    if len(set(df.iloc[:,-1]))>1:
        raise Exception('The explanation does not satisfy the RFD')
        

def alg_config_parse(filename='config.yaml'):
    with open(filename, 'r') as fp:
        try:
            alg_dict = yaml.safe_load(fp)
            return alg_dict
        except yaml.YAMLError as exc:
            print(exc)

def set_distance(list1, list2):
    return len(set(list1) ^ set(list2))  


def select_features(feature_scores, level=0.8):
    sorted_scores = sorted(feature_scores, key=lambda x: abs(x[1]), reverse=True)
    total_score = sum([abs(score) for _, score in sorted_scores])

    selected_features = []
    accumulated_score = 0
    for feature, score in sorted_scores:
        accumulated_score += abs(score)
        selected_features.append(feature)
        if accumulated_score >= total_score * level:
            break

    return selected_features


def compute_con_acc(df, instance_value, feature_list):
    # specified_row = df.loc[instance_id]
    specified_row = instance_value

    mask_same_features = (df[feature_list] == specified_row[feature_list]).all(axis=1)
    mask_different_label = (df['pred_target'] != specified_row['pred_target'])
    mask_same_features_and_different_label = mask_same_features & mask_different_label
    
    count = mask_same_features_and_different_label.sum()
    consistency = (df.shape[0]-count) / df.shape[0]
    
    acc = False
    if count==0:
        acc = True
        
    return round(consistency, 3), acc


def compute_recall(df, instance_value, feature_list1, feature_list2):
    
    mask_same_features1 = (df[feature_list1] == instance_value[feature_list1]).all(axis=1)
    mask_same_features2 = (df[feature_list2] == instance_value[feature_list2]).all(axis=1)
    mask_same_features = mask_same_features1 | mask_same_features2
    
    return round(mask_same_features1.sum() / mask_same_features.sum(), 3)



def compute_faithfulness_er(test_df, instance_value, tem_exp, predict_fn):

    new_instance_value = instance_value.copy()
    
    for col in tem_exp:
        possible_replacements = test_df[test_df[col] != instance_value[col]][col].unique()
        
        if len(possible_replacements) == 0:
            print(f"No alternative values available for column {col}. Skipping...")
            continue
        
        new_value = random.choice(possible_replacements)
        new_instance_value[col] = new_value
    
    rest_values = new_instance_value.iloc[:-1]
    mid_idx = len(rest_values) // 2  
    l_tuple = rest_values[:mid_idx]
    r_tuple = rest_values[mid_idx:]
    from certamain.certa.local_explain import get_original_prediction
    prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
    pre_class = np.argmax(prediction)
    
    if new_instance_value.iloc[-1] == pre_class:
        return True
    else:
        return False


def compute_faithfulness_er_fnum(test_df, instance_value, sorted_column_names, predict_fn): 
    
    same_res_num = 0

    tem_exp = sorted_column_names
    new_instance_value = instance_value.copy()
    
    for col in tem_exp:
        possible_replacements = test_df[test_df[col] != instance_value[col]][col].unique()
        if len(possible_replacements) == 0:
            print(f"No alternative values available for column {col}. Skipping...")
            continue
        
        new_value = random.choice(possible_replacements)            
        new_instance_value[col] = new_value
        
    rest_values = new_instance_value.iloc[:-1]
    mid_idx = len(rest_values) // 2  
    l_tuple = rest_values[:mid_idx]
    r_tuple = rest_values[mid_idx:]
    from certamain.certa.local_explain import get_original_prediction
    prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
    pre_class = np.argmax(prediction)

    if new_instance_value.iloc[-1] == pre_class:
         same_res_num += 1
    
    return same_res_num/1


def compute_faithfulness(test_df, instance_value, sorted_name_exp, best_model): 
            
    same_res_num = 0
    tem_exp = sorted_name_exp
    new_instance_value = instance_value.copy()
    
    for col in tem_exp:
        
        possible_replacements = test_df[test_df[col] != instance_value[col]][col].unique()
        
        if len(possible_replacements) == 0:
            print(f"No alternative values available for column {col}. Skipping...")
            continue
        
        new_value = random.choice(possible_replacements)            
        new_instance_value[col] = new_value
        
    prediction = best_model.predict(new_instance_value.iloc[0:-1].values.reshape(1, -1))[0]

    if new_instance_value.iloc[-1] == prediction:
         same_res_num += 1
    
    return same_res_num/1  


