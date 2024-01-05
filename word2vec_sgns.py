"""
Use word2vec to generate embeddings and corresponding medical codes


Authour: Yufeng
"""




import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from generate_sentence import sentence_generator
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import itertools
import random


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
from sklearn.manifold import TSNE

data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Original_data'
raw_data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/NSF HF Dataset/'
cohort_1_root = os.path.join(data_root,'cohort_1')
cohort_23_root = os.path.join(data_root,'cohort_23')
assign_file = os.path.join(raw_data_root,'Original/Cohort Assignment.csv')
file_code_name = ['VAclass','PCT','ICD']




def train_word2vec(document,vector_size,sg = 1):
    model = Word2Vec(sentences = document, vector_size = vector_size, window = 5, min_count = 1, workers = 4,sg = sg)
    word_embed_dict = dict({})
    for idx, key in enumerate(model.wv.key_to_index):
        word_embed_dict[key] = model.wv[key]
    return word_embed_dict



def split_data(data,test_size = 0.3,random_state = 1234):
    train_data,test_data = train_test_split(data,test_size = test_size,random_state = random_state)
    return train_data,test_data

def shuffle_doc(code_doc,seed):
    random.seed(seed)
    code_s = [random.sample(line, len(line)) for line in code_doc]
    return code_s

sentence_gen = sentence_generator(cohort_1_root,cohort_23_root,assign_file)
cohort,VA,CPT,ICD = sentence_gen.process_two_cohorts_with_no_label_mapping()
code_doc = cohort.full_code.tolist()


###############################################################
print(' There are {} records '.format(cohort.shape[0]))
print(' The number of VA code is {} and the number of CPT code is {} and the number of ICD code is {}'.format(len(np.unique(VA)),len(np.unique(CPT)),len(np.unique(ICD))))
print(" Total number of codes is {}".format(len(np.unique(VA))+len(np.unique(CPT))+len(np.unique(ICD))))
###############################################################
phecode = pd.read_csv('../data_temp/Phecode_map_v1_2_icd10cm_beta.csv',encoding = "ISO-8859-1")
HF_codes = list(phecode[phecode['phecode_str'].str.contains("Heart failure")]['icd10cm'])

vector_size = [60,80,100,120,150,200]
sg_ls = [0,1]

# vector_size = [100]
# sg_ls = [1]

save_dir = '../data_temp/'
visual = False
for sg in sg_ls:
    for vec_size in vector_size:
        code_embed_dict = train_word2vec(code_doc,vec_size,sg = sg)
        embedding = np.stack(list(code_embed_dict.values()),axis = 0)
        embedding_names = np.array(list(code_embed_dict.keys()))
        
        
        if sg == 1 and vec_size == 100:
            print("SAVE embeddings as csv and dict")
            np.save(os.path.join(save_dir,'Code_Embed_Dict.npy'), code_embed_dict) 
            np.savetxt(os.path.join(save_dir,"{}_embedding.csv".format('sgns')), embedding, delimiter=",")
            textfile = open(os.path.join(save_dir,"{}_row_names.csv".format('sgns')), "w")
            for i,element in enumerate(embedding_names):
                if i != len(embedding_names)-1:
                    textfile.write(element + ",")
                else:
                    textfile.write(element)
            textfile.close()



            embedding_group_dict = dict.fromkeys(embedding_names, None)
            for name in embedding_names:
                if name in VA:
                    embedding_group_dict[name] = 'VA_drug'
                elif name in CPT:
                    embedding_group_dict[name]= "CPT_Procedure"
                elif name in ICD:
                    if name in HF_codes:
                        embedding_group_dict[name]= 'Heart Failure'       
                    else:
                        embedding_group_dict[name]= 'ICD_diagnosis'


            np.save(os.path.join(save_dir,'Embedding_Group_Dict.npy'), embedding_group_dict) 

            if visual:

                tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                tsne_results = tsne.fit_transform(embedding)

                df = pd.DataFrame(tsne_results,columns = ['one','two'])
                df['names'] = embedding_names
                df['group'] = df['names'].apply(lambda x: embedding_group_dict[x])

                print('********* Draw All medical codes *********')
                scatter_x = np.array(df['one'])
                scatter_y = np.array(df['two'])
                group = np.array(df['group'])
                group_color = {'Heart Failure':'red',
                                'ICD_diagnosis':'green',
                                'CPT_Procedure':'blue',
                                'VA_drug':'orange',}
                sns.set_style("darkgrid", {'axes.grid' : True})
                fig, ax = plt.subplots(figsize=(20,10))
                for g in np.unique(group):
                    ix = np.where(group == g)
                    if g == 'ICD_diagnosis' or g == 'CPT_Procedure' or g == 'VA_drug':
                        alpha = 0.25
                        s = 60
                    else:
                        alpha = 1.0
                        s = 80
                    ax.scatter(scatter_x[ix], scatter_y[ix], c = group_color[g], label = g, s = s,alpha = alpha)
                lgnd = ax.legend(fontsize=30,scatterpoints=1,loc='center left', bbox_to_anchor=(1, 0.5))
                for handle in lgnd.legendHandles:
                    handle.set_sizes([300.0])

                plt.title('Visualization of medical codes in HeartFailure data',fontsize= 40)
                plt.xlabel('t-SNE X embedding',fontsize= 40)
                plt.ylabel('t-SNE Y embedding',fontsize= 40)
                # plt.show()
                plt.savefig('./plot/All_code_embeddings.png', bbox_inches='tight')

                print('********* Draw ICD medical codes *********')
                group_disease = {'J':'Diseases of the respiratory system',
                                'K':'Diseases of the digestive system',
                                'I':'Diseases of the circulatory system',
                                'G':'Diseases of the nervous system',
                                'D':'Diseases of the blood and blood-forming organs',
                                'E':'Endocrine, nutritional and metabolic diseases',
                                'M':'Diseases of the musculoskeletal system and connective tissue',
                                'N':'Diseases of the genitourinary system'
                                }

                group_color = {'J':'red',
                                'K':'blue',
                                'I':'green',
                                'G':'orange',
                                'D':'purple',
                                'E':'brown',
                                'M':'pink',
                                'N':'grey'
                            }
                embedding_group_dict = dict.fromkeys(embedding_names, None)
                for name in embedding_names:
                    if name in ICD:
                        s = name[0]
                        if s in group_disease.keys():
                            embedding_group_dict[name]=name[0].upper()
                        else:
                            embedding_group_dict[name]= 'Other ICD codes'
                    else:
                        embedding_group_dict[name]= 'Unknown'

                df = pd.DataFrame(tsne_results,columns = ['one','two'])
                df['names'] = embedding_names
                df['group'] = df['names'].apply(lambda x: embedding_group_dict[x])
                df_ICD = df[df['group'] != 'Unknown']
                df_ICD = df_ICD[df_ICD['group'] != 'Other ICD codes']
                df_ICD['color'] = df_ICD['group'].apply(lambda x: group_color[x])
                df_ICD['system'] = df_ICD['group'].apply(lambda x: group_disease[x])
                scatter_x = np.array(df_ICD['one'])
                scatter_y = np.array(df_ICD['two'])
                group = np.array(df_ICD['system'])
                group_color = {'Diseases of the respiratory system':'red',
                                'Diseases of the digestive system':'blue',
                                'Diseases of the circulatory system':'green',
                                'Diseases of the nervous system':'orange',
                                'Diseases of the blood and blood-forming organs':'purple',
                                'Endocrine, nutritional and metabolic diseases':'brown',
                                'Diseases of the musculoskeletal system and connective tissue':'pink',
                                'Diseases of the genitourinary system':'grey'
                            }

                fig, ax = plt.subplots(figsize=(20,10))
                sns.set_style("darkgrid", {'axes.grid' : True})
                for g in np.unique(group):
                    ix = np.where(group == g)
                    ax.scatter(scatter_x[ix], scatter_y[ix], c = group_color[g], label = g, s = 80)
                lgnd = ax.legend(fontsize=50,scatterpoints=1,loc='center left', bbox_to_anchor=(1, 0.5))
                for handle in lgnd.legendHandles:
                    handle.set_sizes([300.0])
                plt.title('Visualization of ICD codes in HeartFailure data',fontsize= 40)
                plt.xlabel('t-SNE X embedding',fontsize= 40)
                plt.ylabel('t-SNE Y embedding',fontsize= 40)
                plt.savefig('./plot/ICD_embeddings.png', bbox_inches='tight')
                
                






        embed_dist = sklearn.metrics.pairwise.cosine_similarity(embedding,embedding)
        embed_dist_df = pd.DataFrame(embed_dist,columns = embedding_names,index =embedding_names) 
        phecode = pd.read_csv('../data_temp/Phecode_map_v1_2_icd10cm_beta.csv',encoding ='latin1')
        intersect_icd_code = set(phecode.icd10cm).intersection(set(embed_dist_df.columns))
        phecode_index = dict(zip(intersect_icd_code,np.arange(len(intersect_icd_code))))
        p = np.array([code for code in phecode.icd10cm if (code in intersect_icd_code) ])
        p_phecode = np.array([phecode[phecode['icd10cm'] == code]['phecode'].values for code in p])
        pair_phecode_list = list(itertools.permutations(p_phecode, 2))
        pair_phecode_value = [ 1 if len(set(pair[0]).intersection(set(pair[1]))) == (len(pair[0]) + len(pair[1]) - 1) else 0 for pair in pair_phecode_list]
        pair_code_list = list(itertools.permutations(p,2))


        pair_phecode_value = np.array(pair_phecode_value)
        pair_cosine_prob = np.array([embed_dist_df.loc[pair[0],pair[1]] for pair in pair_code_list])

        pair_phecode_value = np.array(pair_phecode_value)
        y = pair_phecode_value 
        pred = pair_cosine_prob
        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        auc = metrics.auc(fpr, tpr)
        aucpr = sklearn.metrics.average_precision_score(y, pred)
        if sg == 1:
            method = 'skip_gram'
        elif sg == 0:
            method = 'CBOW'

        string = '\t auc is {:.2f}, aucpr is {:.2f} at dimension {:.2f} with {}'.format(auc,aucpr,vec_size,method)
        print(string)

