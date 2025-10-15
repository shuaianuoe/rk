# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import storge_tree as st
import srk as sc
import pickle
import time
from xgboost import XGBClassifier
from utils import alg_config_parse, compute_con_acc, compute_recall

alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
epsilon = alg_dict['epsilon']
model_num = alg_dict['model_num']
sample_num = alg_dict['sample_num']
window_size = alg_dict['window_size']
step_size = alg_dict['step_size']
strategy = alg_dict['strategy']

print("*"*20)
print("dataset:", datasetsname)
print("step_size:", step_size)

data_df = pd.read_csv('data_process/'+datasetsname+'_test.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
# y_test = data_df['Target']
data_df = data_df.drop('Target', axis=1)
X_test = data_df.iloc[:, :-1]

splits_X = np.array_split(X_test, model_num)
predictions = []
for i in range(model_num):
    print("current model num:", i+1)
    model_file_path = 'results/model/'+datasetsname+'_xgb_model_'+str(i+1)+'.json'
    best_model = XGBClassifier()
    best_model.load_model(model_file_path) 
    pred = best_model.predict(splits_X[i])
    predictions.append(pd.DataFrame(pred))
        
y_pred = pd.concat(predictions, ignore_index=True)
data_df = pd.concat([X_test, y_pred], axis=1)
data_df.columns.values[-1] = 'pred_target'

data_df['is_duplicated'] = data_df.duplicated(data_df.columns[:-1].tolist(), keep=False)
data_df['Y_duplicated'] = data_df.duplicated(subset=data_df.columns[-1:].tolist(), keep=False)
to_drop = data_df[data_df['is_duplicated'] & ~data_df['Y_duplicated']]
data_df = data_df.drop(to_drop.index)
data_df = data_df.drop(['is_duplicated', 'Y_duplicated'], axis=1)
data_df.reset_index(drop=True, inplace=True)

#%% sliding window
batch_size = len(data_df) // model_num
window_size = int(batch_size * window_size)
step_size = int(window_size * step_size)

batches = [data_df.iloc[i*batch_size:(i+1)*batch_size] for i in range(model_num)]

# Create data for sliding windows
# windows = [data_df.iloc[i:i+window_size] for i in range(0, len(data_df) - window_size + 1, step_size)]
windows = [data_df.iloc[i:i+window_size] for i in range(0, len(data_df) - window_size + 1, step_size)]
if len(data_df) % step_size != 0:
    windows.append(data_df.iloc[-window_size:])
    
# num = sample_num // model_num
# selected_indexes = [batch.sample(n=num, random_state=0).index for batch in batches]
# selected_indices = np.concatenate(selected_indexes)

np.random.seed(42) 
selected_indices = np.random.choice(data_df.index[:int(len(data_df)*0.99)], size=sample_num, replace=False)

#%% Start calculating the result of SRK based on window  
with open('data_process/'+datasetsname+'_xgb.pkl', 'rb') as f:
    res_dict = pickle.load(f)
    
tree_set_dict = res_dict['tree_set_dict']
complement_index_dict = res_dict['complement_index_dict']
same_set_dict = res_dict['same_set_dict']
res_dict.clear()

columns_name_list = data_df.columns.values.tolist()   
X = columns_name_list[0:-1]
Y = columns_name_list[-1]

class_names = np.unique(data_df[Y])

res_dict = {}
s_time = np.zeros(sample_num, dtype='float')
exp_s = np.zeros(sample_num, dtype='int')
consistency_s = np.zeros(sample_num, dtype='float')
acc_s = np.zeros(sample_num, dtype='bool')
recall_s = np.zeros(sample_num, dtype='float')

for instance_id in selected_indices:
    print("instance number", np.where(selected_indices == instance_id)[0][0])
    instance_value = data_df.loc[instance_id]
    
    if strategy in ['last']:
        for i, window in reversed(list(enumerate(windows))):
            if instance_id in window.index:
                latest_window = window
                latest_window_index = i
                break  
    if strategy in ['first']:
        for i, window in list(enumerate(windows)):
            if instance_id in window.index:
                latest_window = window
                latest_window_index = i
                break  
    for i, batch in enumerate(batches):
        if instance_id in batch.index:
            instance_batch = batch
            instance_batch_index = i
            break
    
    subsets = {}
    # Calculate complement on the spot
    diff_set_dict = st.get_completary(tree_set_dict, complement_index_dict, same_set_dict, columns_name_list, instance_value)
    universe = set(latest_window.index[latest_window.iloc[:,-1] != instance_value[Y]])
    for x in X:
        subsets[x] = set(diff_set_dict[x][instance_value[x]])
     
    start_time = time.time() * 1000
    cover_sets = sc.greedy_set_cover(universe, subsets, epsilon)
    
    res_dict[instance_id] = cover_sets
    exp_s[np.where(selected_indices == instance_id)[0][0]] = len(cover_sets)  
    s_time[np.where(selected_indices == instance_id)[0][0]] = time.time() * 1000 - start_time
    
    consistency, acc = compute_con_acc(instance_batch, instance_value, cover_sets)
    consistency_s[np.where(selected_indices == instance_id)[0][0]] = consistency
    acc_s[np.where(selected_indices == instance_id)[0][0]] = acc
    
    recall_flag = 1
    if recall_flag==1:
        universe = set(instance_batch.index[instance_batch.iloc[:,-1] != instance_value[Y]])
        for x in X:
            subsets[x] = set(diff_set_dict[x][instance_value[x]])
         
        index_set = set(instance_batch.index)
        for x in X:
            subsets[x] = subsets[x].intersection(index_set)
        real_cover_sets = sc.greedy_set_cover(universe, subsets, epsilon)
        
        recall = compute_recall(instance_batch, instance_value, cover_sets, real_cover_sets)
        recall_s[np.where(selected_indices == instance_id)[0][0]] = recall
        
    
print("*"*20)
print("size")
print("SRK: mean size:", np.mean(exp_s))
print("SRK: mean size:", [round(split.mean(), 3) for split in np.array_split(exp_s, model_num)])

print("*"*20)
print("precision")
print("SRK: mean precision:", np.nanmean(consistency_s))
print("SRK: mean precision:", [round(np.nanmean(split), 3) for split in np.array_split(consistency_s, model_num)])

print("*"*20)
print("conformity")
print("SRK: mean conformity:", np.mean(acc_s))
print("SRK: mean conformity:", [round(split.mean(), 3) for split in np.array_split(acc_s, model_num)])

print("*"*20)
print("recall")
print("SRK: mean recall:", np.mean(recall_s))
print("SRK: mean accuracy:", [round(split.mean(), 3) for split in np.array_split(recall_s, model_num)])
