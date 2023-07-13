from trainer import *
import utils
import numpy as np
import os
from sklearn import preprocessing
import pandas as pd


## read data
embeddings_root ='../data_temp'
pooling_methods = 'attention'
# file = os.path.join(embeddings_root,'MLP_EHR_dict_{}.npy'.format(pooling_methods))
# dataset = np.load(file,allow_pickle=True).item()
dataset = load_data_main(pooling_methods,time_unit = 'days')
main_features = dataset['main_features']
code_num = main_features.shape[1]
labels = dataset['labels']
patient_ID = dataset['patient_ID']
lab_features = dataset['lab_features']
time_elapse = dataset['time_elapse']
split_method = 'patient_wise'
num_classes = 2
random_state = 1995
n_folds = 10
search_iters = 100
n_folds_hyper_tuning = 3
import platform
if platform.system() == 'Windows':
    n_jobs = 1
else:
    n_jobs = -1

out_path = './output/'
model_path = './models/cv'

def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)            
    return sklearn.metrics.roc_auc_score(truth, pred, average=average)

ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
X, X_scaler, X_elapse, y,pat, X_test, X_scaler_test, X_elapse_test,y_test,_ = utils.split_dataset(ss_train_test, main_features, lab_features, \
                                                                                time_elapse,labels, patient_ID,split_method, index=0)

# parameters
# batch_size = 32
weighted_loss = [1,2]
max_steps=2500
# lr=0.003
# regu = 1
# print('The training data size is {} and the test data size is {}'.format(X.shape[0],X_test.shape[0]))
# print('batch size is {} \nweighted_loss is {} \nmax_steps is {} \nlearning rate is {} \nregu is {}'.format(batch_size,weighted_loss,max_steps,lr,regu))

# lr_ls = [0.003,0.001,0.03,0.01]
# bs_ls = [64]
# regu_ls = [10,1,0.1,0.01,0.001] 
lr_ls = [0.03]
bs_ls = [64]
regu_ls = [0.1] 
norm_method_ls = ['all','alpha','None']
row_name_list = []
row_list = []
for lr in lr_ls:
    for batch_size in bs_ls:
        for regu in regu_ls:
            for norm_method in norm_method_ls:
                print('batch size is {} \nweighted_loss is {} \nmax_steps is {} \nlearning rate is {} \nregu is {} and norm method is {}'.format(batch_size,weighted_loss,max_steps,lr,regu,norm_method))

                fold_train = np.zeros([n_folds, 7])
                fold_test = np.zeros([n_folds, 7])
                fold_classifiers = []
                roc_values = {'fpr_test': [],
                            'tpr_test': [],
                            'auc_test': []}
                aucpr_values = {'recall_test': [],
                            'precision_test': [],
                            'aucpr_test': []}
                
                
                
                
                ss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.20, random_state=random_state)
                for index in range(n_folds):
                    print('##### training fold {} #####'.format(index))
                    X_train, X_scaler_train,X_train_elapse, y_train,_, X_val, X_scaler_val,X_val_elapse,y_val,_ = utils.split_dataset(ss, X, X_scaler, \
                                                                                    X_elapse,y,pat,split_method, index=index)
                    # scale lab values
                    scaler = preprocessing.StandardScaler().fit(X_train_elapse.reshape(-1,1))
                    X_train_elapse = scaler.transform(X_train_elapse.reshape(-1,1))
                    X_val_elapse = scaler.transform(X_val_elapse.reshape(-1,1))

                    classifier = generic_LTR(n_classes=2,
                                batch_size = batch_size,
                                weighted_loss = weighted_loss,
                                split_method = 'patient_wise',
                                report_freq = 100,
                                patience_step=500,
                                max_steps=max_steps,
                                learning_rate=lr,
                                regu=regu,
                                random_state = random_state)
                    classifier.fit(X_train, X_scaler_train,X_train_elapse, y_train,code_num, norm_method)
                    evaluation_name = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'F1', 'AUC', 'AUCPR']
                    _, train_metrics, _, _, _, _, _,_,_,_ = utils.cal_acc_full(classifier, X_train,X_scaler_train, X_train_elapse, y_train, num_classes>2)
                    _, test_metrics, _, _, _, fpr_test, tpr_test,precision_p_test, recall_p_test,_ = utils.cal_acc_full(classifier, X_val,X_scaler_val,X_val_elapse,y_val, num_classes>2)
                    fold_train[index,:] = np.array(train_metrics)
                    fold_test[index,:] = np.array(test_metrics)
                    
                    roc_values['fpr_test'].append(fpr_test)
                    roc_values['tpr_test'].append(tpr_test)
                    roc_values['auc_test'].append(test_metrics[5])

                    aucpr_values['recall_test'].append(recall_p_test)
                    aucpr_values['precision_test'].append(precision_p_test)
                    aucpr_values['aucpr_test'].append(test_metrics[-1])
                    fold_classifiers.append(classifier) 
                print(['accuracy',  'recall', 'specificity', 'precision','f1', 'auc', 'aucpr'])
                print('# ============== train ==============#')
                print(np.nanmean(fold_train,axis = 0))
                print(np.nanstd(fold_train,axis = 0))
                print('# ============== test ==============#')
                print(np.nanmean(fold_test,axis = 0))
                print(np.nanstd(fold_test,axis = 0))
                evaluation_name = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'F1', 'AUC', 'AUCPR']
                colnames = ['{}_{}'.format(set_name, eval_name) for set_name in ['Test', 'Train'] for eval_name in evaluation_name]
                show_value_list = []
                show_value_list = utils.show_metrics(fold_test, show_value_list)  
                show_value_list = utils.show_metrics(fold_train, show_value_list)
                eval_series = pd.Series(show_value_list, index=colnames)
                row_name_list.append('model_lr_{}_bs_{}_alpha_regu_{}_method_{}'.format(lr,batch_size,regu,norm_method))
                row_list.append(eval_series)
                pickle.dump(fold_classifiers, open(os.path.join(model_path, 'model_lr_{}_bs_{}_alpha_regu_{}_method_{}.mdl'.format(lr,batch_size,regu,norm_method)), 'wb'))

eval_table = pd.concat(row_list, axis=1).transpose()
eval_table.index = row_name_list
eval_table.to_csv(os.path.join(out_path, 'cv_results_specified_norm_method.csv'.format(random_state)))