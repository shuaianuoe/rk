# -*- coding: utf-8 -*-

import random
import math
from random import choice

def generate_subsets(array,n):
    if n == -1:
        return [[]]
    else:
        subsets = generate_subsets(array,n-1)
        new_subsets = list(subsets)
        for s in list(subsets):
            s= list(s)
            s.append(array[n])
            new_subsets.append(s)
    return new_subsets

def get_all_subsets(n):
    input_set = [x for x in range(1,n+1)]
    return generate_subsets( input_set,len(input_set)-1)

def get_sample_subsets(subsets,n,probability):
    sample_subsets = []
    for s in subsets:
        if len(s)==0 or len(s)>=n-1:
            continue
        elif random.random() < probability:
            sample_subsets.append(s)
    return sample_subsets

def get_elements_from_subsets(subsets):
    elements = set()
    for key in subsets:
        elements = elements.union(subsets[key])
    return elements

#calculate the maximum k for ws such that 2^(-k) < 1/m
def findK(value):
    k = 0
    tmp = math.pow(2,k)
    while (tmp>value):
        k = k-1
        tmp = math.pow(2,k)
    return abs(k)

def initialize_weights_new(subsets, k):
    sample_subsets_weight = {key: math.pow(2,-k) for key, _ in subsets.items()}
   
    return sample_subsets_weight;

#calculate wj for a given element 
def cal_wj(sample_subsets, sample_subsets_weight, current_insid):

    wj = 0
    for each_set, set_val in sample_subsets.items():
        # print(each_set)
        # print(set_val)
        if current_insid in set_val:
            wj += sample_subsets_weight[each_set]
    return wj

# find the subset who covers wj
def find_sj(sample_subsets, current_insid):
    candidate_feature = [k for k, v in sample_subsets.items() if current_insid in v]
    return candidate_feature

# check the resuly is correct
def check_correct(sample_subsets, final_subsets):

    covered_ele = set()
    for fea_num in final_subsets:
        covered_ele.update(sample_subsets[fea_num])
    
    return covered_ele

# initialize the randsc algorithm
def ini_rand_sc(data_df, X, Y, sample_subsets):
    
    final_subsets = []
    cover_elements = set()

    # Start algorithm
    # Step A: Initialization and w_s
    k = findK(1/len(X))
    sample_subsets_weight = initialize_weights_new(sample_subsets, k)  

    # Step B: Pick each s ∈ F with probability ws to C. 
    for feature_name in X:
        if random.random() < sample_subsets_weight[feature_name]:
            final_subsets.append(feature_name)
            cover_elements.update(sample_subsets[feature_name])
    
    return final_subsets, cover_elements, sample_subsets_weight


def rand_sc(epsilon, current_insid, universe_set, cover_elements, sample_subsets, sample_subsets_weight, final_subsets):
    
    while(len(universe_set-cover_elements)>epsilon*len(universe_set)):
        wj = cal_wj(sample_subsets, sample_subsets_weight, current_insid)
        # Step C.2: add to C an arbitrary single set from Sj 
        if wj > math.log(len(universe_set), 2):
            tmp_candidate_feature = find_sj(sample_subsets, current_insid)
            candidate_feature = [x for x in tmp_candidate_feature if x not in final_subsets]
            feature_name = choice(candidate_feature)
            final_subsets.append(feature_name)
            cover_elements.update(sample_subsets[feature_name])
            assert current_insid in cover_elements
        # Step C.3: ej is uncovered and ωj ≤ lg j),
        else:
            # print("aa")
            tmp_candidate_feature = find_sj(sample_subsets, current_insid)
            for tmp_fea in tmp_candidate_feature:
                # print(tmp_fea)
                if sample_subsets_weight[tmp_fea] < 1:
                    sample_subsets_weight[tmp_fea] *=2
            candidate_feature = [x for x in tmp_candidate_feature if x not in final_subsets]
            for tmp_fea in candidate_feature:
                if random.random() < sample_subsets_weight[tmp_fea]:            
                    final_subsets.append(tmp_fea)
                    cover_elements.update(sample_subsets[tmp_fea])
            
    if epsilon == 0:
        if check_correct(sample_subsets, final_subsets) != universe_set:
            raise Exception('The result does not cover universe')
        
    return final_subsets
    
