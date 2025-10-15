# -*- coding: utf-8 -*-

import sys
sys.path.append("..")


import os
import pickle
import warnings
import pandas as pd
from certa.models.utils import get_model
from certa.explain import CertaExplainer
from certa.utils import merge_sources
from certa.local_explain import get_original_prediction
import numpy as np

from utils import alg_config_parse

# Encountered a missing en_core_web_lg error in the command line, so downloading en_core_web_lg here.
import spacy.cli
spacy.cli.download("en_core_web_lg")


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

# %% 

alg_dict = alg_config_parse('../config.yaml')            
dataset = alg_dict['datasetsname']
sample_num = alg_dict['sample_num']
datadir = 'datasets/'+dataset
modeldir = 'model/ditto/' + dataset + '/model.pkl'  

model = load_model(modeldir)
if model is None:
    model = get_model('ditto', modeldir, datadir, dataset)
    save_model(model, modeldir)


lsource = pd.read_csv(datadir+'/tableA.csv')
rsource = pd.read_csv(datadir+'/tableB.csv')


test = pd.read_csv(datadir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

if  os.path.exists('../data_process/'+dataset+'_test.csv'):
    print("prediction file is found!")
    test_df = pd.read_csv('../data_process/'+dataset+'_test.csv')
else:
    y_pred_list = []
    for idx in range(test_df.shape[0]):
        print("generating prediction file!")
        rand_row = test_df.iloc[idx]
        l_id = int(rand_row['ltable_id'])
        l_tuple = lsource.iloc[l_id]
        r_id = int(rand_row['rtable_id'])
        r_tuple = rsource.iloc[r_id]
        prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
        class_to_explain = np.argmax(prediction)
        y_pred_list.append(class_to_explain)
        
    test_df['pred_target'] = y_pred_list
    test_df.to_csv('../data_process/'+dataset+'_test.csv', index=False)

