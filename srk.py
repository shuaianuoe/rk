# -*- coding: utf-8 -*-
from itertools import chain, combinations
from collections import defaultdict


def powerset(iterable):
    """Calculate the powerset of any iterable.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def optimal_set_cover(universe, subsets):
    """ Optimal algorithm - DONT USE ON BIG INPUTS - O(2^n) complexity!
    Finds the minimum cost subcollection os S that covers all elements of U
    Args:
        universe (list): Universe of elements
        subsets (dict): Subsets of U {S1:elements,S2:elements}
    """
    best_set = []
    # for copy
    subsets_copy = subsets.copy() 
    
    pset = powerset(subsets_copy.keys())
    
    for subset in pset:
        
        covered = set()
        for s in subset:
            covered.update(subsets_copy[s])
        #if len(covered) == len(universe):
        if len(universe.difference(covered))==0:
            best_set = subset
            break
        
    return list(best_set)


def greedy_set_cover(universe, subsets, epsilon):
    """Approximate greedy algorithm for set-covering. Can be used on large
    inputs - though not an optimal solution.
    

    Args:
        universe (set): Universe of elements
        subsets (dict): Subsets of U {S1:elements,S2:elements}
        
    """
    rem_num = len(universe)*epsilon
    cover_sets = []
    # for copy
    subsets_copy = subsets.copy() 
    
    uni_set = set(universe)
    while len(uni_set)>rem_num: # 0:
    #    print("uni_set length", len(uni_set))
        max_inter = -1
        max_set = None
        for s, elements in subsets_copy.items():
            tmp_max_inter = len(uni_set.intersection(elements))
            if tmp_max_inter > max_inter:
                max_inter = tmp_max_inter
                max_set = s
                max_elements = elements
        # delete elements from uni_set
        uni_set.difference_update(max_elements)
        subsets_copy.pop(max_set, None)
        cover_sets.append(max_set)
        
    return cover_sets

def greedy_linear_set_cover(subsets):
    """Approximate greedy algorithm for set-covering. Can be used on large
    inputs - though not an optimal solution.
    
    Note this is a linear implementation.

    Args:
        subsets (dict): Subsets of U {S1:elements,S2:elements}
        
    """
    # First prepare a list of all sets where each element appears
    D = defaultdict(list)
    F = subsets.copy()
    
    F = list(F.values())
    
    for y, S in enumerate(F):
        for a in S:
            D[a].append(y)
     
    L = defaultdict(set)        
    # Now place sets into an array that tells us which sets have each size
    for x,S in enumerate(F):
        L[len(S)].add(x) 
    
    E = [] # Keep track of selected sets
    
    # Now loop over each set size
    for sz in range(max(len(S) for S in F),0,-1):
        if sz in L:
            P = L[sz] # set of all sets with size = sz
            while len(P):
                x = P.pop()
                E.append(x)
                for a in F[x]:
                    for y in D[a]:
                        if y!=x:
                            S2 = F[y]
                            L[len(S2)].remove(y)
                            S2.remove(a) 
                            L[len(S2)].add(y)
      
    return E         


def completary_set(data_df, columns_name_list):
    """Compute the complementary set based on each x in columns_name_list

    Args:
        data_df (dataframe): indexs must begin from 0
        columns_name_list (list): the candidate column names
    """
    instance_num = data_df.shape[0]

    
    diff_set_dict = {}
    
    for feature_name in columns_name_list:
        # print("feature_name", feature_name)
        diff_set_dict[feature_name] = {}
        temp_data_ser = data_df[feature_name]
        # get the distict values based on current feature
        dis_set = set(temp_data_ser)
        # compute the complement of the instance index corresponding to each value
        for value in dis_set:
            f = temp_data_ser==value
            indices = temp_data_ser[f].index
            # all the instance indexs
            instance_set = set(i for i in range(instance_num))
            diff_set_dict[feature_name][value] = instance_set.difference(set(indices))

    return diff_set_dict


def completary_oneins_set(data_df, columns_name_list, instance_value):
    """Compute the complementary set based on each x in columns_name_list
       Note that this only supply to one instance

    Args:
        data_df (dataframe): indexs must begin from 0
        columns_name_list (list): the candidate column names
    """
    instance_num = data_df.shape[0]

    
    diff_set_dict = {}
    
    for feature_name in columns_name_list:
        # print("feature_name", feature_name)
        diff_set_dict[feature_name] = {}
        temp_data_ser = data_df[feature_name]
        value = instance_value[feature_name]
        f = temp_data_ser==value
        indices = temp_data_ser[f].index
        # all the instance indexs
        instance_set = set(i for i in range(instance_num))
        diff_set_dict[feature_name][value] = instance_set.difference(set(indices))
        

    return diff_set_dict


def key_for_distinctY(data_df, Y, res_greedy_dict):
    """get the distinct values based on current feature

    """
    temp_data_ser = data_df[Y]
    dis_Y_set = set(temp_data_ser)
    minkey_dict = {}   
    for label in dis_Y_set:
        tmp_res = set()
        indices = temp_data_ser[temp_data_ser==label].index
        for index in indices:
            tmp_res.update(res_greedy_dict[index])
        minkey_dict[label] = list(tmp_res)
    
    return minkey_dict


def values_ins(data_df, res_dict):
    """get the specific data values based on res_dict

    """
    values_dict = {}
    for instance_id in range(data_df.shape[0]):
        tmp_ser = data_df.loc[instance_id][res_dict[instance_id]]
        dd = defaultdict(list)
        values_dict[instance_id]=tmp_ser.to_dict(dd)
        
    return values_dict


def counts_ins(data_df, values_dict):
    
    count_dict = {}
    
    for instance_id in range(data_df.shape[0]):
        tmp_keys = values_dict[instance_id].keys()
        
        df = data_df.copy()
        for key in tmp_keys:
            df = df.loc[(df[key] == values_dict[instance_id][key])]
        
        count_dict[instance_id] = df.shape[0]
    
    return count_dict


def check_exp(data_df, beexplain_id, final_subsets):
    
    beexplain_value = data_df.loc[beexplain_id]
    df = data_df.copy()
    for key in final_subsets:
        df = df.loc[(df[key] == beexplain_value[key])]
    if len(set(df.iloc[:,-1]))>1:
        # raise Exception('The explanation does not satisfy the key')
        return 0
    return 1
