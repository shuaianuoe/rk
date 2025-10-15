# -*- coding: utf-8 -*-
import redis
import pandas as pd
import json
from utils import alg_config_parse

alg_dict = alg_config_parse('config.yaml')            
datasetsname = alg_dict['datasetsname']
print("*"*20)
print("dataset:", datasetsname)

data_df = pd.read_csv('data_process/'+datasetsname+'_test.csv', index_col=0)
data_df.reset_index(drop=True, inplace=True)
data_df = data_df.drop('Target', axis=1)

#%%  The first scenario for srk
r = redis.Redis(host='localhost', port=6379, db=0)
# Convert dataframe to JSON and store it in Redis
r.set('my_df', data_df.to_json())
# Read data from Redis and then switch back to dataframe
df_back = pd.read_json(r.get('my_df').decode('utf-8'))

# r.flushdb()

#%%  The second scenario for orsk and ssrk: Insert the data in the dataframe line by line into Redis, and read the latest line inserted into Redis
for _, row in data_df.iterrows():
    r.lpush('my_df1', json.dumps(row.to_dict()))
# Read the latest row of data in Redis
latest_row = pd.read_json(r.lindex('my_df1', 0).decode('utf-8'), typ='series')
r.connection_pool.disconnect()