
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import copy
import os
import sklearn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
import utils
import pickle
from load_data import *


def build_dataset_loader(features, scalers, times,labels = None, batch_size = 4, infinite = False):
 
    if labels is not None:
        tensor_features = torch.from_numpy(features).float()
        tensor_scalers = torch.from_numpy(scalers).float()
        tensor_time = torch.from_numpy(times).float()
        tensor_labels = torch.from_numpy(labels.astype(np.int32))
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device), tensor_scalers.to(device),tensor_time.to(device),tensor_labels.to(device))
    else:
        tensor_features = torch.from_numpy(features).float()
        tensor_scalers = torch.from_numpy(scalers).float()
        tensor_time = torch.from_numpy(times).float()
        dataset = torch.utils.data.TensorDataset(tensor_features.to(device),tensor_scalers.to(device),tensor_time.to(device))
    
    if infinite:
        data_loader = utils.repeater(torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=True,drop_last=True))
    else:
        data_loader = torch.utils.data.DataLoader(dataset, int(batch_size), shuffle=False)
        
    return features, data_loader

class Net(nn.Module):
    def __init__(self,n_classes,code_dim = 100,activation = torch.nn.RReLU(),scaler_dim = 4):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.code_dim = code_dim
        self.scaler_dim = scaler_dim
        self.activation = activation
        self.dim_weight = Parameter(torch.Tensor(self.code_dim,1))
        self.w = Parameter(torch.Tensor(self.code_dim + self.scaler_dim,2))
    def forward(self,x,x_scaler = None,x_time = None):
        """
        x is a tensor : 400 patient * 8000 code * 100 dim
        """
        x_copy = x.clone()
        alpha = torch.permute(x.matmul(self.dim_weight),(0,2,1)) # 400 * 8000 * 1   --> 400 * 1 * 8000
        # alpha = (torch.tanh(alpha) + 1)/2
        alpha = torch.tanh(alpha)
        x = torch.squeeze(torch.matmul(alpha,x)) # 400 * 100 
        if len(x.shape) == 1:
          x = x[None, :]
        if x_scaler is not None:
            x = torch.cat([x,x_scaler],dim = -1)
        if x_time is not None:
            x = x + utils.positional_encoding(x_time, x.shape[1], n=10000)
        # w = torch.tanh(self.w)
        x = torch.matmul(x,self.w) # 400 * 2
        # x = torch.matmul(x,self.w)
        self.code_weight = alpha
        # beta = x_copy.matmul(self.w[:self.code_dim,1])
        beta = x_copy.matmul(self.w[:self.code_dim,1])
        return x,alpha,beta
    def reset_parameters(self):   
        nn.init.kaiming_uniform_(self.dim_weight.data,nonlinearity='relu') 
        nn.init.kaiming_uniform_(self.w.data,nonlinearity='relu') 

        # nn.init.uniform_(self.dim_weight.data,a = -1,b =1) 
        # nn.init.uniform_(self.w.data,a = 0.1,b =1) 


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Personalized_LTR(BaseEstimator, ClassifierMixin): 
    def __init__(self, 
                 n_classes=2,
                 batch_size = 8,
                 weighted_loss = [1,1.5],
                 split_method = 'patient_wise',
                 report_freq=50,
                 patience_step=500,
                 max_steps=10000,
                 learning_rate=0.001,
                 regu=0,
                 random_state = 1995
                 ):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.weighted_loss = weighted_loss
        self.split_method = split_method
        self.report_freq = report_freq
        self.patience_step = patience_step
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.regu = regu
        self.random_state = random_state
        self.loss_records = []

    def fit(self, X_train, X_scaler,X_time,y_train):
        torch.manual_seed(self.random_state)
        
        
        _, train_loader = build_dataset_loader(X_train, X_scaler,X_time, y_train, self.batch_size, infinite=True)
        _, train_loader_for_eval = build_dataset_loader(X_train, X_scaler,X_time, y_train, self.batch_size,infinite=False)

        net = Net(self.n_classes)
        net.reset_parameters()
        net.to(device)
        if self.weighted_loss is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weighted_loss, dtype=dtype, device=device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)


        for global_step, (inputs, scaler, time, labels) in enumerate(train_loader, 0):
            labels = labels.type(torch.long)
            optimizer.zero_grad()
            if global_step>7000:
                for g in optimizer.param_groups:
                    g['lr'] = self.learning_rate*0.5
            out,alpha,beta = net(inputs,scaler,time)
            cross_entropy = criterion(out, labels)
            
            for param in net.parameters():
                regu_loss = torch.norm(param,1)
                
            # neg_pen_loss = self._negative_penalty_loss(alpha,beta)
            # loss = cross_entropy + self.regu * regu_loss + self.regu * neg_pen_loss
            loss = cross_entropy + self.regu * regu_loss

            loss.backward() 
            optimizer.step()
            self.loss_records.append(loss)
            if (global_step+1) % self.report_freq == 0:
                _, _, _, _, f1_train, auc_train, aucpr_train, _, _ = self._model_testing(net, train_loader_for_eval)
                print(f'Step {global_step}, train_loss: {loss:.3f}, l2_loss: {regu_loss:.3f}, main_loss: {cross_entropy:.3f}, train_auc: {auc_train:.3f}, train_aucpr: {aucpr_train:.3f}, train_f1: {f1_train:.3f}.')
            if global_step > self.max_steps:
                break

        self.estimator = net
        self.classes_ = unique_labels(y_train)

        return self

    def predict(self, X, scaler, time):
        check_is_fitted(self)
        
        _, test_loader = build_dataset_loader(X, scaler, time, batch_size=self.batch_size, infinite=False)
            
        pred_list = []
        for i, (inputs, scaler, time) in enumerate(test_loader, 0):
            out,alpha,beta = self.estimator(inputs, scaler, time)
            pred = torch.argmax(out, dim=1)
            pred_list.append(pred)
        pred_list = np.concatenate(pred_list, axis=0)
        return pred_list

    def predict_proba(self, X, scaler, time):
        check_is_fitted(self)
        _, test_loader = build_dataset_loader(X, scaler, time, batch_size=self.batch_size, infinite=False)
        prob_list = []
        weight_list = []
        for i, (inputs, scaler, time) in enumerate(test_loader, 0):
            out,alpha,beta = self.estimator(inputs, scaler, time)
            prob = F.softmax(out, dim=1)
            prob_list.append(prob.detach())  
            weight_list.append(alpha.detach())      
            
        prob_list = np.concatenate(prob_list, axis=0)
        prob_list = np.round(prob_list, 3)

        weight_list = np.concatenate(weight_list, axis=0)
        weight_list = np.round(weight_list, 3)
        return prob_list,weight_list

    def _model_testing(self, net, test_loader):

        pred_list = []
        label_list = []
        prob_list = []
        for i, (inputs, scaler, time, labels) in enumerate(test_loader, 0):
            labels = labels.type(torch.long)
            x,alpha,beta = net(inputs,scaler, time)

                                
            prob = F.softmax(x, dim=1)
            pred = torch.argmax(x, dim=1)
            pred_list.append(pred)
            label_list.append(labels)
            prob_list.append(prob.detach())
        
        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        prob_list = torch.cat(prob_list, dim=0)
            
        pred = pred_list.numpy()
        labels = label_list.numpy()
        probs = prob_list.numpy()
        probs = np.round(probs, 3)
        
        acc = np.sum(pred == labels)/len(labels)
        sen = np.sum(pred[labels==1])/np.sum(labels)
        spe = np.sum(1-pred[labels==0])/np.sum(1-labels)
        pre = np.sum(pred[labels==1])/(np.sum(pred)+1)   
        f1 = sklearn.metrics.f1_score(labels, pred)
        # auc, aucpr = 0,0
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:,1], pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucpr = sklearn.metrics.average_precision_score(labels, probs[:,1])
        return acc, sen, spe, pre, f1, auc, aucpr, probs, labels
    
    
    def _negative_penalty_loss(self,alpha,beta):

        return torch.mean(torch.log(1 + torch.exp(-torch.multiply(torch.squeeze(alpha),beta))))
        





if __name__ == '__main__':
    ## read dataset ## 
    embeddings_root ='../data_temp'
    pooling_methods = 'attention'
    file = os.path.join(embeddings_root,'MLP_EHR_dict_{}.npy'.format(pooling_methods))
    # dataset = np.load(file,allow_pickle=True).item()
    dataset = load_data_main('attention',time_unit = 'days')
    main_features = dataset['main_features']
    labels = dataset['labels']
    patient_ID = dataset['patient_ID']
    lab_features = dataset['lab_features']
    time_elapse = dataset['time_elapse']
    split_method = 'patient_wise'
    num_classes = 2
    random_state = 1995

    ss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    # X_train,y_train,pat_train,X_test, y_test,pat_test = utils.simple_split_data(ss_train_test, main_features, labels, patient_ID, split_method, index=0)
    X_train, X_scaler, X_elapse, y_train, pat, X_test, X_scaler_test, X_elapse_test,y_test,_ = utils.split_dataset(ss_train_test, main_features, lab_features, \
                                                                                time_elapse,labels, patient_ID,split_method, index=0)


    # parameters
    batch_size = 64
    weighted_loss = [1,2]
    max_steps=3000
    lr=0.003
    regu = 1
    exp_id = 'pos_neg'
    
    print('The training data size is {} and the test data size is {}'.format(X_train.shape[0],X_test.shape[0]))
    print('Experiment: {} ; batch size is {} \nweighted_loss is {} \nmax_steps is {} \nlearning rate is {} \nregu is {}'.format(exp_id,batch_size,weighted_loss,max_steps,lr,regu))



    classifier = Personalized_LTR(n_classes=2,
                 batch_size = batch_size,
                 weighted_loss = weighted_loss,
                 split_method = 'patient_wise',
                 report_freq = 500,
                 patience_step=500,
                 max_steps=max_steps,
                 learning_rate=lr,
                 regu=regu,
                 random_state = random_state)
    classifier.fit(X_train, X_scaler, X_elapse,y_train)
    loss_records = classifier.loss_records
    evaluation_name = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'F1', 'AUC', 'AUCPR']
    print(evaluation_name)
    _, train_metrics, _, _, _, _, _,_,_,train_weights = utils.cal_acc_full(classifier, X_train,X_scaler, X_elapse, y_train, num_classes>2)
    print('# ============== train ==============#')
    print(train_metrics)
    _, test_metrics, _, _, _, _, _,_,_,test_weights = utils.cal_acc_full(classifier, X_test,X_scaler_test, X_elapse_test,y_test, num_classes>2)
    print('# ============== test ==============#')
    print(test_metrics)
    
    
    
    
    # save model
    d = 'Feb_23th'
    save_path = os.path.join('./models/','train_test')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    pickle.dump(classifier, open(os.path.join(save_path, f'model_{d}_{exp_id}_bs_{batch_size}_lr{lr}_regu_{regu}_steps{max_steps}_timeaware.mdl'), 'wb'))
    
    with open(os.path.join(save_path, f'model_{d}_{exp_id}_bs_{batch_size}_lr{lr}_regu_{regu}_steps{max_steps}_loss_records.p'), 'wb') as f:
        pickle.dump(loss_records,f)

    # save weights
    weights_path = os.path.join('./models/','weights')
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    np.save(os.path.join(weights_path,f'weight_{d}_{exp_id}_bs_{batch_size}_lr_{lr}_regu_{regu}_steps_{max_steps}_train.npy'), train_weights)
    np.save(os.path.join(weights_path,f'weight_{d}_{exp_id}_bs_{batch_size}_lr_{lr}_regu_{regu}_steps_{max_steps}_test.npy'), test_weights)

    



