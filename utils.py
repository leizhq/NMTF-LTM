import math
import csv
import numpy as np
from scipy import sparse
import pandas as pd
import os
import random


def outputCSV(file_dir, output_list):
    with open(file_dir, "a", newline='') as fobj:
        writer = csv.writer(fobj)
        for row in output_list:
            writer.writerow(row)


def compute_TU(topic_word, N):
    """
    :param topic_word: topic_word matrix
    :param N: top word count
    :return: average TU for the whole matrix
    """
    topic_size, word_size = np.shape(topic_word)
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    TU = 0
    cnt = [0 for i in range(word_size)]
    for topic in topic_list:
        for word in topic:
            cnt[word] += 1
    for topic in topic_list:
        TU_t = 0
        for word in topic:
            TU_t += 1/cnt[word]
        TU_t /= N
        TU += TU_t

    TU /= topic_size

    return TU


def compute_TU_list(topic_word, N):
    """
    :param topic_word: topic_word matrix
    :param N: top word count
    :return: TU for each individual topic
    """
    topic_size, word_size = np.shape(topic_word)
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    TU = []
    cnt = [0 for i in range(word_size)]
    for topic in topic_list:
        for word in topic:
            cnt[word] += 1
    for topic in topic_list:
        TU_t = 0
        for word in topic:
            TU_t += 1/cnt[word]
        TU_t /= N
        TU.append(TU_t)

    return TU


def print_topic_word(word_list, save_dir, topic_word, N):
    # print top N words of each topic
    topic_size, vocab_size = np.shape(topic_word)

    with open(save_dir, 'a', encoding='utf-8') as fout:
        print('-------------------- Topic words --------------------', file=fout)
        for topic_idx in range(topic_size):
            top_word_list = []
            print('['+str(topic_idx)+'] ', end='', file=fout)
            top_word_idx = np.argsort(topic_word[topic_idx, :])[-N:]
            for i in range(N):
                top_word_list.append(word_list[top_word_idx[i]])

            # print words
            for word in top_word_list:
                print(word, ' ', end='', file=fout)
            print('\n', end='', file=fout)
        print('\n', end='', file=fout)

    print('save done!')


def save_as_triple(Y, filename):
    Y_coo = sparse.coo_matrix(Y)
    result_matrix = open(filename, 'w')
    result_matrix.writelines('row_idx,col_idx,data')
    # Y_coo = Y.tocoo()
    for i in range(len(Y_coo.data)):
        result_matrix.writelines("\n" + str(Y_coo.row[i]) + "," + str(Y_coo.col[i]) + "," + str(Y_coo.data[i]))


def read_triple(filename, get_sparse=False):
    tp = pd.read_csv(open(filename))
    rows, cols, data = np.array(tp['row_idx']), np.array(tp['col_idx']), np.array(tp['data'])
    if get_sparse:
        return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1))
    return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1)).toarray()


# freeze random seeds
def freeze_seed(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
