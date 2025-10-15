# -*- coding: utf-8 -*-
"""
This script is our space optimization strategy: 
pre process an instance in a window and construct a tree for the feature values of each feature for easy access in the future.
"""

# get index set with the same value for each distinct value 
same_set_dict = {}
# subset for saving storage by useing recursion. the inner value is the distinct value
tree_set_dict = {}
# temp tree_set_dict
tree_tmp_set_dict = {}
# get the complement index for each distinct value  
complement_index_dict={}

def merge(left, right, feature_name):
    result = []
    i, j = 0, 0
    while i < len(left) :
        result.append(left[i])
        i += 1
    while j < len(right) :
        result.append(right[j])
        j += 1
    
    tree_tmp_set_dict[feature_name][str(left)] = left
    tree_tmp_set_dict[feature_name][str(right)] = right

    return result

# recursion to split the list into sub-lists, fisrt step to save storage
def mergeTree(L, feature_name):
    if len(L) < 2:
        return L
    else:
        mid = len(L) // 2
        left = mergeTree(L[:mid], feature_name)
        right = mergeTree(L[mid:], feature_name)
        return merge(left, right, feature_name)


def forin(a, b):
    for obj in b:
        if obj in a:
            return 1
    return 0


def completary_index(data_df, columns_name_list):
  
    for feature_name in columns_name_list:
    
            # print("feature_name", feature_name)
            same_set_dict[feature_name] = {}
            tree_tmp_set_dict[feature_name] = {}
            tree_set_dict[feature_name] = {}
            complement_index_dict[feature_name] = {}
    
            
            # ==== set same_set_dict. around 20 mins for 1M(0.5M)
            temp_data_ser = data_df[feature_name]
            # get the distinct values based on current feature
            dis_set = set(temp_data_ser)
            num = 0
            for value in dis_set:
                # print("num", num)
                num += 1
                f = temp_data_ser==value
                same_set_dict[feature_name][value] = list(temp_data_ser[f].index)
            # print("aaa")
            # ==== set tree_tmp_set_dict and tree_set_dict. very fast
            mergeTree(list(dis_set), feature_name)
            # change the key name for convenience
            key_name = 0
            for old_key in tree_tmp_set_dict[feature_name].keys():
                tree_set_dict[feature_name][str(key_name)] = tree_tmp_set_dict[feature_name][old_key]
                key_name += 1
            # free the memory
            tree_tmp_set_dict[feature_name].clear()

            # ==== set complement_index_dict
            # sort dictionary key by value list length. very fast
            sort_key_list = sorted(tree_set_dict[feature_name], key=lambda k: len(tree_set_dict[feature_name][k]), reverse=True)
            

            for value in dis_set:
                tmp_index = []
                tmp_set = set([value])
                # print("tmp_set", tmp_set)
                tmp_len = -1
                for key in sort_key_list:
                    key_set = set(tree_set_dict[feature_name][key])

                    # it seems no much improvement
                    if tmp_len>2 and tmp_len==len(key_set):
                        continue
                    # intersection is null, which means this key can be added
                    if len(tmp_set.intersection(key_set)) == 0:
                        tmp_index.append(key)
                        tmp_set.update(key_set)
    
                    # to speed: when the remaining is one element, just find the key from the value
                    if len(tmp_set) == len(dis_set)-1:
                        one_set = dis_set.difference(tmp_set)
                        one_key = list(tree_set_dict[feature_name].keys())[list(tree_set_dict[feature_name].values()).index(list(one_set))]
                        tmp_index.append(one_key) 
                        tmp_set.update(one_set)

                    if len(tmp_set) == len(dis_set): 
                        complement_index_dict[feature_name][value] = tmp_index
                        break
        
    return tree_set_dict, complement_index_dict, same_set_dict 


# we should get the complenment based on  tree_set_dict, complement_index_dict and same_set_dict 
# At first glance, it looks a little circuitous here

def get_completary(tree_set_dict, complement_index_dict, same_set_dict, columns_name_list, instance_value):

    diff_set_dict = {}
    for feature_name in columns_name_list:
        diff_set_dict[feature_name] = {}
        
        key = instance_value[feature_name]
        diff_set_dict[feature_name][key] = []
        values = complement_index_dict[feature_name][key]
     
        # for each storage index(values), we get the same index(index)
        for index in values:
            same_values = tree_set_dict[feature_name][index]
            # for each same index, we get the specific complement
            for same_index in same_values:
                diff_set_dict[feature_name][key].extend(same_set_dict[feature_name][same_index])

    return diff_set_dict



