import os
import sys
import configparser as cp
import numpy as np
from scipy import sparse

from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix
# from sklearn.cluster import KMeans
from coclust.clustering import SphericalKmeans

import utils


work_directory = os.path.dirname(os.path.abspath(__file__))
exp_ini = sys.argv[1]

# experiment.ini
exp_config = cp.ConfigParser()
exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')
data_name = exp_config.get(exp_ini, 'data_name')
model_name = exp_config.get(exp_ini, 'model_name')
# dataset.ini
data_config = cp.ConfigParser()
data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
filename_prefix = data_config.get(data_name, 'filename_prefix')
chunknum = int(data_config.get(data_name, 'chunk_num'))
n_clusters = len(data_config.get(data_name, 'label').split(','))

data_dir = os.path.join(work_directory, "Dataset", data_name)
res_dir = os.path.join(work_directory, "res", data_name, model_name, exp_ini)

label_file_list = [os.path.join(data_dir, filename_prefix+'_'+str(i+1)+'_label') for i in range(chunknum)]
if model_name == 'PNMTF-LTM' or model_name == 'NMTF-LTM':
    res_file_list = [os.path.join(res_dir, 'V_'+str(i)+'.csv') for i in range(chunknum)]
else:
    res_file_list = [os.path.join(res_dir, 'DT_' + str(i) + '.csv') for i in range(chunknum)]


def get_true_label(filename_list):
    res = []
    label_id_mapping = {}
    label_id_count = 0
    
    for filename in filename_list:
        with open(filename) as fobj:
            for line in fobj.readlines():
                label = line.strip()
                if label in label_id_mapping:
                    res.append(label_id_mapping[label])
                else:
                    res.append(label_id_count)
                    label_id_mapping[label] = label_id_count
                    label_id_count += 1
    return np.array(res)


def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def get_f_measure(labels_true, labels_pred, beta=5.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return f_beta


def cal_indexes(labels_true, labels_pred):
    """
    RI, NMI, F_beta, Purity
    """
    ri = rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    f_beta = get_f_measure(labels_true, labels_pred)
    purity = accuracy(labels_true, labels_pred)
    return nmi, ri, f_beta, purity


if __name__ == '__main__':
    input_matrices = []
    for filename in res_file_list:
        input_matrices.append(utils.read_triple(filename))
    # doc-topic
    X = np.concatenate(input_matrices, axis=0)
    X = sparse.csr_matrix(X)
    print(X.shape)

    out_file = os.path.join(res_dir, 'eval_clustering_new.csv')
    if not os.path.exists(out_file):
        utils.outputCSV(out_file, [['NMI', 'RI', 'F_beta', 'Purity']])

    labels_true = get_true_label(label_file_list)

    for i in range(5):
        model = SphericalKmeans(n_clusters=n_clusters, max_iter=20, weighting=False)  # weighting: perform TF-IDF inside
        model.fit(X)
        labels_pred = model.labels_

        nmi, ri, f_beta, purity = cal_indexes(labels_true, labels_pred)
    
        print('NMI:', nmi, 'RI:', ri, 'F_beta', f_beta, 'Purity', purity)
        
        utils.outputCSV(out_file, [[nmi, ri, f_beta, purity]])
