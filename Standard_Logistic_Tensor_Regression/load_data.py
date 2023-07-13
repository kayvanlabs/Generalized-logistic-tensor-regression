
"""
Main functions for load data
Created on Fri Sep 2 2022
@author: Yufeng Zhang
"""




import numpy as np
from numpy import genfromtxt
import csv
import os
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
import copy


import sys
sys.path.append("../Data_prep")
# sys.path.append("../data_temp")
from generate_sentence import sentence_generator

root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/'
raw_data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/NSF HF Dataset/'
data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Original_data'
save_root = os.path.join(root,'Data/Processed/Yufeng/Original_data')

cohort_1_root = os.path.join(data_root,'cohort_1')
cohort_23_root = os.path.join(data_root,'cohort_23')
assign_file = os.path.join(raw_data_root,'Original/Cohort Assignment.csv')
embeddings_root ='../data_temp'

HT_code = ['33940','00580','33944','33945','33975','33976','33977','33978','33981','02YA0Z0','33979','33980','33982','33983','93750','02HA0QZ',
            'I97.0','I97.110','I97.120','I97.130','I97.190','T82.897','T86.298','Z76.82','Z95.811','02YA0Z1','02WA0QZ','02PA0QZ','33289','93264','33993']


def read_lab_values():
    cohort_numerical_value = pd.read_csv(os.path.join(data_root,'cohort123_lab_corrected_48hw_nosd.csv'))
    select_scaler_names = ['HFID', 'EncID','first48h_BP_mean','first48h_HeartRate_mean','first48h_CHLOR','first48h_SOD','EncounterStart','EncounterEnd','Cohort']
    cohort_numerical_value = cohort_numerical_value.loc[:,select_scaler_names]
    return cohort_numerical_value

def read_code_sentence_ls(cohort_1_root,cohort_23_root,assign_file):
    sentence_gen = sentence_generator(cohort_1_root,cohort_23_root,assign_file)
    cohort,_,_,_ = sentence_gen.process_two_cohorts_with_no_label_mapping()
    return cohort

def convert_gender_to_int(x):
    return 0 if x == 'M' else 1

def date_convert(date_str):
    try:
        d = datetime.strptime(date_str, '%m/%d/%y %H:%M').date()
        return d
    except TypeError:
        return np.nan

def date_convert_noseconds(date_str):
    try:
        d = datetime.strptime(date_str, '%m/%d/%y').date()
        return d
    except TypeError:
        return np.nan



def Global_Average_embeddings(word_index,select_embeddings,sentence_ls):
    pooling_embeddings = []
    for i,single_sentence in enumerate(sentence_ls):
        code_name_index = []
        for k in single_sentence:
            code_name_index.append(word_index[k])
        single_embeddings = np.nanmean(np.take(select_embeddings,code_name_index,axis = 0),axis = 0)
        pooling_embeddings.append(single_embeddings)
    pooling_embeddings = np.stack(pooling_embeddings,axis = 0)
    return pooling_embeddings

def Tensor_embeddings(word_index,full_embeddings,sentence_ls):
    pooling_embeddings = []
    mat_len = range(full_embeddings.shape[0])
    for i,single_sentence in enumerate(sentence_ls):
        ls = single_sentence.copy()
        index_array = np.array([word_index[word] for word in ls])
        index_array = np.array([i for i in mat_len if i not in index_array])
        select_X = copy.deepcopy(full_embeddings)
        select_X[np.ix_(index_array)] = 0
        pooling_embeddings.append(select_X.tolist())
    pooling_embeddings = np.stack(pooling_embeddings,axis = 0)
    return pooling_embeddings


def read_embeddings_and_names(embeddings_root):
    
    
    embedding = genfromtxt(os.path.join(embeddings_root,'sgns_embedding.csv'), delimiter=',')
    print("The shape of embeddings:",embedding.shape)
    
    # read embedding names
    embedding_names = []
    with open(os.path.join(embeddings_root,'sgns_row_names.csv'), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            embedding_names.extend(line)
    embedding_names = np.array(embedding_names)

    word_index =  dict(zip(embedding_names, range(len(embedding_names))))
            
    # select embeddings with name appeared at least once in medical histories
    select_embeddings = []
    for word,index in word_index.items():
        select_embeddings.append(embedding[np.where(embedding_names == word)[0],:])
    select_embeddings = np.squeeze(np.stack(select_embeddings,axis = 0))
    
    return embedding,select_embeddings,word_index


def load_data_main(pooling_methods = 'global_average',save_file = True, time_unit = 'days'):
    cohort_numerical_value = read_lab_values()
    Cohort = read_code_sentence_ls(cohort_1_root,cohort_23_root,assign_file)

    # merge
    Full_data = pd.merge(Cohort,cohort_numerical_value,on = ['HFID', 'EncID'],how='left')
    # filter out corhot 3 and those records without labels
    Full_data  = Full_data[Full_data ['Cohort'] != 3]
    Full_data  = Full_data[Full_data['Cohort'].notnull()]
    print('The average length of medical sentences of label 1 and 2: {}'.format(int(np.mean([len(i) for i in Full_data.full_code.tolist()]))))
    # switch column order to make label at the last column
    Full_data  = Full_data [[c for c in Full_data  if c != 'Cohort'] + ['Cohort']]
    # convert date
    Full_data.EncounterStart = Full_data.EncounterStart.apply(lambda x: date_convert(x))
    print('The numebr of records with label 1 and 2 are : ',Full_data.shape[0])


    # select patients with at least two visits
    uniq_ID = set(np.unique(Full_data['HFID']))
    ID_ENC = {}
    ID_CNT = {}
    for idx in uniq_ID:
        ID_ENC[idx] = list(np.unique(Full_data[Full_data['HFID'] == idx]['EncID']))
        ID_CNT[idx] = len(list(np.unique(Full_data[Full_data['HFID'] == idx]['EncID'])))
    id_ls = []
    for idx in ID_ENC:
        if len(ID_ENC[idx]) >= 2:
            id_ls.append(idx)
    
    # based on the select patient ID id_ls to build a new dataset
    colnames = ['first_visit_' + i for i in Full_data.columns]
    colnames = [c for c in colnames if 'Cohort' not in c]
    
    new_tbl = pd.DataFrame()
    unorder_ID = []
    for idx in id_ls:
        sub = Full_data.loc[Full_data.HFID == idx,:]
        l = list(sub.EncounterStart)
        if all(l[i] <= l[i+1] for i in range(len(l) - 1)) is False:
            sub = sub.sort_values(by=['EncounterStart'], ascending = True)
            unorder_ID.append(idx)
        cnt = sub.shape[0]
        for i in range(cnt-1):
            if sub.iloc[i,1] == sub.iloc[i+1,1]:
                continue
            else:
                sub_sub = pd.DataFrame(sub.iloc[i,:-1]).transpose()
                sub_sub.columns = colnames
                sub_sub['Final_label'] = sub.iloc[i+1,-1]
                sub_sub['last_visit_EncounterStart'] = sub.iloc[i+1,-3]
                new_tbl = pd.concat([new_tbl,sub_sub],ignore_index=True)
    first_sentence_ls = new_tbl.first_visit_full_code.tolist()
    
    new_first_sentence_ls = []
    for ls in first_sentence_ls:
        new_ls =  [d for d in ls if not (d.startswith('Z95') or d.startswith('Z94') or d.startswith('Z98') or d in HT_code)]
        new_first_sentence_ls.append(new_ls)
        
        
    first_sentence_ls = new_first_sentence_ls
    
    print('The number of sentences:',len(first_sentence_ls))

    
    # Calculate elapsed time
    time_elapse = new_tbl.last_visit_EncounterStart - new_tbl.first_visit_EncounterStart
    if time_unit == 'months':
        time_elapse = np.round(np.array(time_elapse.dt.days)/30,1)
    elif time_unit == 'days':
        time_elapse = np.array(time_elapse.dt.days)
    
    
    
    
    
    
    
    
    
    
    # embeddings generation
    full_embeddings, select_embeddings,word_index = read_embeddings_and_names(embeddings_root)
    if pooling_methods == 'global_average':
        first_embeddings = Global_Average_embeddings(word_index,select_embeddings,first_sentence_ls)
    elif pooling_methods == 'attention':
        first_embeddings = Tensor_embeddings(word_index,full_embeddings,first_sentence_ls)
    

    # for new dataset select corresponding lab values
    select_scaler_names = ['first48h_BP_mean','first48h_HeartRate_mean','first48h_CHLOR','first48h_SOD']
    first_scaler_names = ['first_visit_' + i for i in select_scaler_names]
    lab_scalers = np.array(new_tbl[first_scaler_names])
    
    ################################################
    #============    data for analysis ============#
    
    # impute missing values in lab with multiple imputation
    imp = SimpleImputer(missing_values = np.nan, strategy = 'median')
    imp.fit(lab_scalers)
    SimpleImputer()
    # lab values
    lab_features = imp.transform(lab_scalers)
    # main features
    main_features = first_embeddings
    # patient ID
    patient_ID = np.array(new_tbl.first_visit_HFID)

    # label processing 
    labels = np.array(new_tbl['Final_label'])
    labels = labels - 1
    where_0 = np.where(labels == 0)
    where_1 = np.where(labels == 1)

    labels[where_0] = 1
    labels[where_1] = 0
    print('Pos sample : {}  and Neg samples: {}'.format(sum(labels == 1), sum(labels == 0)))

    

    print('***********************************************************************')
    print('Statistics for time elapse: the min is {} and the max is {} days.'.format(min(time_elapse),max(time_elapse)))
    print('main_features shape:',main_features.shape)
    print('lab_features shape:',lab_features.shape)
    print('time_elapse shape:',time_elapse.shape)
    print('pos shape:',sum(labels == 1))
    print('neg shape:',sum(labels == 0))
    print('labels shape:',labels.shape)
    print('patient_ID shape:',patient_ID.shape)
    print('total patient:',len(np.unique(patient_ID)))
    print('***********************************************************************')
    dataset = {'main_features':main_features,
                'lab_features':lab_features,
                'time_elapse':time_elapse,
                'labels':labels,
                'patient_ID':patient_ID}
    if save_file:
        np.save(os.path.join(embeddings_root,'MLP_EHR_dict_{}.npy'.format(pooling_methods)), dataset) 

    return dataset

if __name__ == '__main__':
    dataset = load_data_main(pooling_methods = 'global_average',save_file = True) #'global_average','Graph','attention'



