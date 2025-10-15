# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import storge_tree as st
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils import alg_config_parse

alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
multiclass = alg_dict['multiclass']
print("*"*20)
print("dataset:", datasetsname)

data_df = pd.read_csv('data_process/'+datasetsname+'_data.csv')
data_df.columns = list(data_df.columns[:-1]) + ['Target']

# %% first process the data and encode their features.
label_encoders = {}
for column in data_df.columns:
    le = LabelEncoder()
    data_df[column] = le.fit_transform(data_df[column])
    label_encoders[column] = le


#%% Divide the training set and test set (70% -30%). Train Set to Train Model
train, test = train_test_split(data_df, test_size=0.3, random_state=42)

# Assuming the last column is a label
X_train = train.iloc[:, :-1]  
y_train = train.iloc[:, -1]  

X_test = test.iloc[:, :-1]    
y_test = test.iloc[:, -1]     

# Specify the file path to save or load the model
model_file_path = 'results/model/'+datasetsname+'_xgb_model.json'

# Check if the model file exists
if os.path.exists(model_file_path):
    # Load the model from the file
    print('Loading the existing model...')
    best_model = XGBClassifier()
    best_model.load_model(model_file_path)
else:
    # Create and train the model
    print('Training a new model...')
    params = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
        'colsample_bylevel': [0.5, 0.7, 1],
        'min_child_weight': [0.5, 1, 2, 5],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0],
        'scale_pos_weight': [1, 2, 3]
    }
    # This is to address the imbalance of multiple categories
    weights = np.ones(y_train.shape)
    if multiclass==True:
        print("This is a multiclass dataset")
        # multi-class, only for airbnb dataset
        model = XGBClassifier(objective='multi:softprob', num_class=10, n_estimators=3000)
        random_search  = RandomizedSearchCV(model, param_distributions=params, n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, verbose=3,random_state=42)
        weights[y_train == 9] = 0.14
    else:
        model = XGBClassifier(n_estimators=1000)
        random_search  = RandomizedSearchCV(model, param_distributions=params, n_iter=30, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3,random_state=42)
    random_search.fit(X_train, y_train, sample_weight=weights)
    # Get the best estimator
    best_model = random_search.best_estimator_
    # Save the model to a file
    best_model.save_model(model_file_path)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

train.to_csv('data_process/'+datasetsname+'_train.csv')
test['pred_target'] = y_pred
test.to_csv('data_process/'+datasetsname+'_test.csv')




#%% Calculate a tree in advance and store the complement. Convenient for subsequent calculations, note that this will not affect the algorithm results
test = test.drop('Target', axis=1)
columns_name_list = test.columns.values.tolist() 
X = columns_name_list[0:-1]
Y = columns_name_list[-1]

dup_df = test[test.duplicated(subset=X, keep=False)]
if dup_df.shape[0]>0:
    # find those duplicate rows index
    dup_list = dup_df.groupby(X).apply(lambda x: tuple(x.index)).tolist()
    for i in range(len(dup_list)):
        # here is loc, because we don't reset index before
        tmp_dup_df = dup_df.loc[list(dup_list[i])]
        # Y has more than 1 distinct values, which means the same X showing the different labels
        if len(set(tmp_dup_df.iloc[:,-1]))>1:
            test = test.drop(tmp_dup_df.index)
            # print(i)
            # raise Exception('There is the same X showing the different labels.')

test = test.reset_index(drop=True)

tree_set_dict, complement_index_dict, same_set_dict  = st.completary_index(test, columns_name_list)

res_dict = {}
res_dict['tree_set_dict'] = tree_set_dict
res_dict['complement_index_dict'] = complement_index_dict
res_dict['same_set_dict'] = same_set_dict

with open('data_process/'+datasetsname+'_xgb.pkl', 'wb') as f:
    pickle.dump(res_dict, f)


#%% In the monitor interpretation experiment, based on the training set, it is divided into several parts, each training one xgb model with different parameters.

#This module can be run separately. Not related to the above

from xgboost import XGBClassifier
import random

alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
model_num = alg_dict['model_num']

print("*"*20)
print("dataset:", datasetsname)

data_df = pd.read_csv('data_process/'+datasetsname+'_train.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
X_train = data_df.iloc[:, :-1]  
y_train = data_df.iloc[:, -1]   

# Specify the file path to save or load the model
model_file_path = 'results/model/'+datasetsname+'_xgb_model.json'
print('Loading the existing model...')
best_model = XGBClassifier()
best_model.load_model(model_file_path)
params = best_model.get_params()


max_depth_range = [3, 4, 5, 6, 7, 8, 9, 10]
learning_rate_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
subsample = [0.6,0.7,0.8,0.9,1]
gamma = [0.8,0.9,1]
param_list = []

for _ in range(model_num):
    new_params = params.copy()
    new_params['max_depth'] = random.choice(max_depth_range)
    new_params['learning_rate'] = random.choice(learning_rate_range)
    new_params['subsample'] = random.choice(subsample)
    new_params['gamma'] = random.choice(gamma)
    param_list.append(new_params)

splits_X = np.array_split(X_train, model_num)
splits_y = np.array_split(y_train, model_num)

models = []
for i in range(model_num):
    print("training model num:", i+1)
    model = XGBClassifier(**param_list[i])
    model.fit(splits_X[i], splits_y[i])
    models.append(model)
    model.save_model('results/model/'+datasetsname+'_xgb_model_'+str(i+1)+'.json')   



