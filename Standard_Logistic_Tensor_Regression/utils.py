import numpy as np
import copy
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from itertools import repeat
import torch
from sklearn.utils.fixes import loguniform


def positional_encoding(pos,d = 100, n=10000):
    seq_len = pos.shape[0]
    P = np.zeros((seq_len, d))
    for l,k in enumerate(pos):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[l, 2*i] = np.sin(k/denominator)
            P[l, 2*i+1] = np.cos(k/denominator)
    return torch.from_numpy(P).float()



# def positional_encoding_3d(pos,d = 100, n=10000): # time_elapse, X_t.shape[1], n = 10000
#     seq_len = pos.shape[0]
#     P = np.zeros((seq_len, d))
#     for l,k in enumerate(pos):
#         for i in np.arange(int(d/2)):
#             denominator = np.power(n, 2*i/d)
#             P[l, 2*i] = np.sin(k/denominator)
#             P[l, 2*i+1] = np.cos(k/denominator)
#     return P

def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)

def split_by_id(id_record,test_size,random_state):
    uniq_id = np.unique(id_record)
    id_train,id_test = train_test_split(uniq_id,test_size = test_size,random_state = random_state)
    return id_train,id_test


def split_dataset(stratified_split, data, scaler_data, time_elapse,labels, patient_ID, split_method, index=0):        

    if split_method == 'patient_wise':   
        uids = copy.deepcopy(patient_ID) 
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==0]).difference(uids_HT_VAD_set)
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.zeros(uids_too_well_arr.shape[0])], axis=0)
    
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)

        train_idx = [True if i in uids_train else False for i in patient_ID]
        test_idx = [True if i in uids_test else False for i in patient_ID]


        X_train = data[train_idx,:]
        X_scaler_train = scaler_data[train_idx,:]
        X_train_elapse = time_elapse[train_idx]
        y_train = labels[train_idx]
        pat_train = patient_ID[train_idx]
        
     
        X_test = data[test_idx,:]
        X_scaler_test = scaler_data[test_idx,:]
        X_test_elapse = time_elapse[test_idx]
        y_test = labels[test_idx]
        pat_test = patient_ID[test_idx]

        
    elif split_method == 'sample_wise':
        index = list(stratified_split.split(data, labels))[index]
        train_index = index[0]
        test_index = index[1]

        X_train, X_test = np.take(data, train_index, axis=0), np.take(data, test_index, axis=0)
        y_train, y_test = np.take(labels, train_index, axis=0), np.take(labels, test_index, axis=0)
    
    else:
        raise NotImplementedError
         

         
    return X_train.astype(np.float64), X_scaler_train.astype(np.float64),X_train_elapse.astype(np.float64),y_train.astype(np.int64),pat_train.astype(np.int64), \
        X_test.astype(np.float64), X_scaler_test.astype(np.float64),X_test_elapse.astype(np.float64),y_test.astype(np.int64),pat_test.astype(np.int64)


def indices_to_one_hot(data, n_classes=None):
    data = data.astype(np.int32)
    if n_classes is None:
        n_classes = np.max(data) + 1
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def cal_acc_full(model, features, scaler, time, labels,multiple):
    if multiple:
        average_method = 'macro'
    else:
        average_method = 'binary'
    probs,weight_list = model.predict_proba(features,scaler,time)   #features,scaler,time
    predictions = np.argmax(probs, axis=-1)
    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method)
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = 2*precision*recall/(precision + recall + 0.1)
    if multiple:
        aucpr = sklearn.metrics.average_precision_score(indices_to_one_hot(labels), 
                                                        probs, average=average_method)
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels, np.max(labels)+1), probs,
                                            average=average_method, multi_class='ovr')
        fpr, tpr = None
    else:
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels), probs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1],drop_intermediate = False)
        precision_p, recall_p, _ = sklearn.metrics.precision_recall_curve(labels, probs[:,1])

    metrics = np.array([accuracy, recall, specificity, precision, f1, auc, aucpr])
    metrics_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr']
    return predictions, metrics, metrics_name, probs, labels, fpr, tpr,precision_p, recall_p,weight_list
    





def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def generate_mean_metrics(tprs_list, fprs_list, aucs_list, type):
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs_list = []
    if type == 'auc':
        for i in range(len(aucs_list)):
            interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
            interp_tpr[0] = 0.0
            interp_tprs_list.append(interp_tpr)
        mean_tpr = np.mean(interp_tprs_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    if type == 'aucpr':
        for i in range(len(aucs_list)):
            interp_tpr = np.interp(mean_fpr, fprs_list[i], tprs_list[i])
            interp_tpr[0] = 1.0
            interp_tprs_list.append(interp_tpr)
        mean_tpr = np.mean(interp_tprs_list, axis=0)
        mean_tpr[-1] = 0.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    
    return mean_fpr, mean_tpr, mean_auc


def draw_AUC(auc_dict,color_items,out_path,random_state):
    #-------------------------------ROC-----------------------------------
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    for i, key in enumerate(auc_dict):
        model_name = key
        roc_values = auc_dict[key]
        mean_fpr, mean_tpr, mean_auc = generate_mean_metrics(roc_values['tpr_test'], roc_values['fpr_test'], roc_values['auc_test'],type = 'auc')
        ax.plot(mean_fpr, mean_tpr, color=color_items[i][-1],label=model_name,lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title=f"ROC curve")
        ax.legend(loc="lower right")
        plt.savefig(os.path.join(out_path, 'ROC_{}_ML.png'.format(random_state)), dpi=300)

def draw_AUPRC(aucpr_dict,color_items,out_path,random_state):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    for i, key in enumerate(aucpr_dict):
        model_name = key
        aucpr_values = aucpr_dict[key]
        mean_recall, mean_precision, _ = generate_mean_metrics(aucpr_values['recall_test'], aucpr_values['precision_test'], aucpr_values['aucpr_test'],type = 'aucpr')
        ax.plot(mean_recall, mean_precision, color=color_items[i][-1],label=model_name,lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title=f"AUPRC curve")
        ax.legend(loc="lower right")
    plt.savefig(os.path.join(out_path, 'AUCPR_{}_ML.png'.format(random_state)), dpi=300)





def simple_split_data(stratified_split, data, labels, patient_ID, split_method, index=0): # ss_train_test, main_features, labels,patient_ID, split_method, index=0
    if split_method == 'patient_wise':   
        uids = copy.deepcopy(patient_ID) 
        uids_HT_VAD_set = set(uids[labels==1])
        uids_too_well_set = set(uids[labels==0]).difference(uids_HT_VAD_set)
        uids_HT_VAD_arr = np.array(list(uids_HT_VAD_set))
        uids_too_well_arr = np.array(list(uids_too_well_set))
        
        uids = np.concatenate([uids_HT_VAD_arr, uids_too_well_arr], axis=0)
        uids_label = np.concatenate([np.ones(uids_HT_VAD_arr.shape[0]), 
                                     np.zeros(uids_too_well_arr.shape[0])], axis=0)
    
        index = list(stratified_split.split(uids, uids_label))[index]
        
        uids_train = np.take(uids, index[0], axis=0)
        uids_test = np.take(uids, index[1], axis=0)

        train_idx = [True if i in uids_train else False for i in patient_ID]
        test_idx = [True if i in uids_test else False for i in patient_ID]


        X_train = data[train_idx,:]
        y_train = labels[train_idx]
        pat_train = patient_ID[train_idx]

        X_test = data[test_idx,:]
        y_test = labels[test_idx]
        pat_test = patient_ID[test_idx]
        
        return X_train.astype(np.float64),y_train.astype(np.int64),pat_train.astype(np.int64), \
                X_test.astype(np.float64), y_test.astype(np.int64),pat_test.astype(np.int64)
def create_personalized_LTR_grid():
    param_grid = { 'batch_size': [8,16,32],
                  'learning_rate': loguniform(5e-3, 1e-1),
                  'regu': [0,1,3,5],
                }
    
    return param_grid

def show_metrics(metrics, show_value_list=None):
    """ Calculate the average and standard deviation from multiple repetitions and format them.
    """
    eval_m, eval_s = np.nanmean(metrics, 0), np.nanstd(metrics,0) 
    for i in range(eval_m.shape[0]):
        show_value_list.append('{:.3f} ({:.3f})'.format(eval_m[i], eval_s[i]))
    return show_value_list


def cal_acc_ML(model, features,labels, multiple):
    
    probs = model.predict_proba(features)  
    predictions = np.argmax(probs, axis=-1)
            # Calculate matrix
    if multiple:
        average_method = 'macro'
    else:
        average_method = 'binary'

    recall = sklearn.metrics.recall_score(labels, predictions, average=average_method)
    precision = sklearn.metrics.precision_score(labels, predictions, average=average_method, zero_division=0)
    specificity = sklearn.metrics.recall_score(1-labels, 1-predictions, average=average_method)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = 2*precision*recall/(precision + recall + 0.1)
    
    if multiple:
        aucpr = sklearn.metrics.average_precision_score(indices_to_one_hot(labels), 
                                                        probs, average=average_method)
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels, np.max(labels)+1), probs,
                                            average=average_method, multi_class='ovr')
        fpr, tpr = None
    else:
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        auc = sklearn.metrics.roc_auc_score(indices_to_one_hot(labels), probs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1],drop_intermediate = False)
        precision_p, recall_p, _ = sklearn.metrics.precision_recall_curve(labels, probs[:,1])

    metrics = np.array([accuracy, recall, specificity, precision, f1, auc, aucpr])
    metrics_name = ['accuracy',  'recall', 'specificity', 'precision',
                    'f1', 'auc', 'aucpr']
    return predictions, metrics, metrics_name, probs, labels, fpr, tpr,precision_p, recall_p






