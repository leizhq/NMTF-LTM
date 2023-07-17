import os
import gc
import sys
import numpy as np
import sklearn.metrics.pairwise as pw

import time
import configparser as cp
from tqdm import tqdm

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

data_dir = os.path.join(work_directory, "Dataset", data_name)
res_dir = os.path.join(work_directory, "res", data_name, model_name, exp_ini)

printAsc = False
recall = [5, 10, 0.02]
sample = False


def get_label(filename):
    with open(filename) as fobj:
        return fobj.read().splitlines()


def compare_labels(train_labels, test_label, label_type="", evaluation_type="", labels_to_count=[]):
    vec_goodLabel = []

    if label_type == "single":
        test_labels = [test_label] * train_labels.shape[0]

        vec_goodLabel = np.array((train_labels == test_labels), dtype=np.int8)
    elif label_type == "multi":
        if not len(train_labels[0]) == len(test_label):
            print("Mismatched label vector length")
            exit()

        test_labels = np.asarray(test_label)
        labels_comparison_vec = np.dot(train_labels, test_labels)

        if evaluation_type == "relaxed":
            vec_goodLabel = np.array((labels_comparison_vec != 0), dtype=np.int8)

        elif evaluation_type == "strict":
            test_label_vec = np.ones(train_labels.shape[0]) * np.sum(test_label)
            vec_goodLabel = np.array((labels_comparison_vec == test_label_vec), dtype=np.int8)

        else:
            print("Invalid evaluation_type value.")

    else:
        print("Invalid label_type value.")

    return vec_goodLabel


def perform_IR_prec(kernel_matrix_test, train_labels, test_labels, list_percRetrieval=None, single_precision=False, label_type="", evaluation="", index2label_dict=None, labels_to_not_count=[], corpus_docs=None, query_docs=None, IR_filename=""):
    '''
    :param kernel_matrix_test: shape: size = |test_samples| x |train_samples|
    :param train_labels:              size = |train_samples| or |train_samples| x num_labels
    :param test_labels:               size = |test_samples| or |test_samples| x num_labels
    :param list_percRetrieval:        list of fractions or number at which IR has to be calculated
    :param single_precision:          True, if only one fraction is used
    :param label_type:                "single" or "multi"
    :param evaluation:                "strict" or "relaxed", only for 
    :return:
    '''

    if not len(test_labels) == len(kernel_matrix_test):
        print('mismatched samples in test_labels and kernel_matrix_test')
        exit()

    labels_to_count = []

    prec = []

    if single_precision:
        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
        prec_num_docs = np.floor(list_percRetrieval[0] * kernel_matrix_test.shape[1])
        vec_simIndexSorted_prec = vec_simIndexSorted[:, :int(prec_num_docs)]
        
        for counter, indices in enumerate(vec_simIndexSorted_prec):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]
            else:
                classQuery = test_labels[counter]
                tr_labels = train_labels[indices]
            list_percPrecision = np.zeros(len(list_percRetrieval))

            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type)

            list_percPrecision[0] = np.sum(vec_goodLabel) / float(len(vec_goodLabel))

            prec += [list_percPrecision]
    else:
        list_totalRetrievalCount = []   # list of number at which IR has to be calculated
        for frac in list_percRetrieval:
            if frac < 1:
                list_totalRetrievalCount.append(int(np.floor(frac * kernel_matrix_test.shape[1])))
            else:
                list_totalRetrievalCount.append(frac)

        if sorted(list_totalRetrievalCount) != list_totalRetrievalCount:
            print("recall is not ascending.")
            exit()
        start = time.time()

        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, :-list_totalRetrievalCount[-1]-1:-1]
        end = time.time()
        print("argsort time:", end-start)
        print("vec_simIndexSorted", time.asctime( time.localtime(time.time()) ))

        for counter, indices in tqdm(enumerate(vec_simIndexSorted)):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]
            else:
                classQuery = test_labels[counter]
                tr_labels = np.array(train_labels)[indices]
            if printAsc: print("choose label", time.asctime( time.localtime(time.time()) ))
            
            # list_percRetrieval: list of fractions at which IR has to be calculated
            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation, labels_to_count=labels_to_count)
            if printAsc: print("compare_labels", time.asctime( time.localtime(time.time()) ))
            
            countGoodLabel = 0
            list_percPrecision = np.zeros(len(list_percRetrieval))
            for indexRetrieval, totalRetrievalCount in enumerate(list_totalRetrievalCount):
                if indexRetrieval == 0:
                    countGoodLabel += np.sum(vec_goodLabel[:int(totalRetrievalCount)])
                else:
                    countGoodLabel += np.sum(vec_goodLabel[int(lastTotalRetrievalCount):int(totalRetrievalCount)])

                list_percPrecision[indexRetrieval] = countGoodLabel / float(totalRetrievalCount)
                lastTotalRetrievalCount = totalRetrievalCount
            
            if printAsc: print("count", time.asctime( time.localtime(time.time()) ))

            prec.append(list_percPrecision)
        print("compare and count", time.asctime( time.localtime(time.time()) ))

    prec = np.mean(prec, axis=0)
    
    del vec_simIndexSorted
    gc.collect()

    return prec


if __name__ == '__main__':

    label_file_list = [os.path.join(data_dir, filename_prefix+'_'+str(i+1)+'_label') for i in range(chunknum)]
    if model_name == 'PNMTF-LTM' or model_name == 'NMTF-LTM':
        res_file_list = [os.path.join(res_dir, 'V_'+str(i)+'.csv') for i in range(chunknum)]
    else:
        res_file_list = [os.path.join(res_dir, 'DT_' + str(i) + '.csv') for i in range(chunknum)]
    label_list = []

    for filename in label_file_list:
        label_list.append(get_label(filename))

    doc_topic_matrices = []
    for filename in res_file_list:
        doc_topic_matrices.append(utils.read_triple(filename))
    rb = []
    for i in range(1, chunknum):
        query_vectors = doc_topic_matrices[i]
        if sample:
            query_vectors = query_vectors[::20]
        query_label = label_list[i]
        if sample:
            query_label = query_label[::20]
        ra = []
        for j in range(i):
            corpus_vectors = doc_topic_matrices[j]
            corpus_label = label_list[j]
            single_precision = False if len(recall) > 1 else True
            start = time.time()
            similarity_matrix = pw.cosine_similarity(query_vectors, corpus_vectors)
            end = time.time()
            print("similarity time, query:%d, train:%d, time:%f" %(i, j, end-start))
            print("similarity_matrix", time.asctime( time.localtime(time.time()) ))
            results = perform_IR_prec(similarity_matrix, corpus_label, query_label, list_percRetrieval=recall, single_precision=single_precision, label_type="single")
            del similarity_matrix
            gc.collect()
            if j == 0:
                ra = [[results[k]] for k in range(len(recall))]
            else:
                for k in range(len(recall)):
                    ra[k].append(results[k])
        if i == 1:
            rb = [[ra[k]] for k in range(len(ra))]
        else:
            for k in range(len(recall)):
                rb[k].append(ra[k])
        print(rb)

    for k in range(len(recall)):
        utils.outputCSV(os.path.join(res_dir, 'eval_IR_P_at_'+str(recall[k]).replace('.', 'p')+'.csv'), rb[k])
