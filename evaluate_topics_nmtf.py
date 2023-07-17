import os
import sys
import time
import csv
import configparser as cp
import numpy as np

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

import utils


def BoW(Dt_path):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase=False)
    Dt_pt = open(Dt_path)
    X = vectorizer.fit_transform(Dt_pt)
    vocabulary = vectorizer.get_feature_names()
    Dt_pt.close()

    return X.toarray(), vocabulary


def texts_corpus_for_eval(pipeline):
    texts = []
    for file_path in pipeline:
        file_pt = open(file_path, 'r')
        for line in file_pt:
            text = line.strip().split()
            texts.append(text)
        file_pt.close()
    corpus = corpora.Dictionary(texts)
    return texts, corpus


def find_top_word(W_t_T_sort_ind, top_n, vocabulary, pipeline_corpus):
    num_word_topic = W_t_T_sort_ind.shape[0]
    top_word = []
    pipeline_corpus_token = list(pipeline_corpus.token2id.keys())
    for i in range(num_word_topic):
        line = []
        for j in W_t_T_sort_ind[i]:
            if vocabulary[j] in pipeline_corpus_token:
                line.append(vocabulary[j])
            if len(line) == top_n:
                break
        top_word.append(line)
    return top_word


def evaluate_coh(texts, corpus, W_t_T, top_n, vocabulary, metrics):
    W_t_T_sort_ind = np.argsort(-W_t_T)

    top_word = find_top_word(W_t_T_sort_ind, top_n, vocabulary, corpus)
    coherence = {}
    methods = metrics
    for method in methods:
        coherence[method] = CoherenceModel(topics=top_word, texts=texts, dictionary=corpus, coherence=method).get_coherence()
    return coherence


def outputCSV(file_dir, output_list):
    with open(file_dir, "w", newline='') as fobj:
        writer = csv.writer(fobj)
        for row in output_list:
            writer.writerow(row)


if __name__ == '__main__':

    work_directory = os.path.dirname(os.path.abspath(__file__))
    exp_ini = sys.argv[1]

    # experiment.ini
    exp_config = cp.ConfigParser()
    exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

    data_name = exp_config.get(exp_ini, 'data_name')
    model_name = exp_config.get(exp_ini, 'model_name')
    top_n = int(exp_config.get(exp_ini, 'top_n'))
    metrics = exp_config.get(exp_ini, 'coh_metrics').split(',')

    # dataset.ini
    data_config = cp.ConfigParser()
    data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')

    filename_prefix = data_config.get(data_name, 'filename_prefix')
    chunknum = int(data_config.get(data_name, 'chunk_num'))

    # dirs
    pipeline = [(filename_prefix + "_{chunk_id}").format(chunk_id=chunk_id) for chunk_id in range(1, chunknum + 1)]
    pipeline = [os.path.join("Dataset", data_name, filename) for filename in pipeline]

    res_dir = os.path.join("res", data_name, model_name, exp_ini)
    print(exp_ini)
    W_file_list = [os.path.join(res_dir, 'W_' + str(i) + '.csv') for i in range(chunknum)]
    V_file_list = [os.path.join(res_dir, 'V_' + str(i) + '.csv') for i in range(chunknum)]
    S_file_list = [os.path.join(res_dir, 'S_' + str(i) + '.csv') for i in range(chunknum)]
    H_file_list = [os.path.join(res_dir, 'H_' + str(i) + '.csv') for i in range(chunknum)]

    # parse pipeline
    texts, corpus = texts_corpus_for_eval(pipeline)
    
    scores_W = []
    scores_WS = []
    TUs_W = []
    TUs_WS = []
    ppxs_W = []
    ppxs_WS = []

    for i in range(chunknum):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('evaluating T =', i)
        
        # read factor matrices
        W_t = utils.read_triple(W_file_list[i])
        S_t = utils.read_triple(S_file_list[i])
        V_t = utils.read_triple(V_file_list[i])
        H_t = utils.read_triple(H_file_list[i])

        # read data matrices
        doc_word, vocabulary = BoW(pipeline[i])
        
        # coherence
        print('evaluating coherence ...')
        score_W = evaluate_coh(texts, corpus, W_t.transpose(), top_n, vocabulary, metrics)
        scores_W.append(score_W)
        print("coherence scores for W: \n", score_W)
        score_WS = evaluate_coh(texts, corpus, (W_t.dot(S_t)).transpose(), top_n, vocabulary, metrics)
        scores_WS.append(score_WS)
        print("coherence scores for WS: \n", score_WS)
        
        # TU
        print('evaluating TU ...')
        TU_W = utils.compute_TU(W_t.transpose(), top_n)
        TUs_W.append(TU_W)
        print("TU for W: ", TU_W)
        TU_WS = utils.compute_TU((W_t.dot(S_t)).transpose(), top_n)
        TUs_WS.append(TU_WS)
        print("TU for WS: ", TU_WS)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    results_W = [metrics+['TU']]
    results_WS = [metrics+['TU']]

    for i in range(chunknum):
        # W
        result_W = [scores_W[i][metric] for metric in metrics]
        result_W.append(TUs_W[i])
        results_W.append(result_W)
        # WS
        result_WS = [scores_WS[i][metric] for metric in metrics]
        result_WS.append(TUs_WS[i])
        results_WS.append(result_WS)

    outputCSV(os.path.join(res_dir, 'topic_eval_W.csv'), results_W)
    outputCSV(os.path.join(res_dir, 'topic_eval_WS.csv'), results_WS)
    
    
    
    

