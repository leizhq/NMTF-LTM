import sys
import configparser as cp
import os
import numpy as np
import pandas as pd


def get_res_list(exp_ini):
    work_directory = os.path.dirname(os.path.abspath(__file__))

    # experiment.ini
    exp_config = cp.ConfigParser()
    exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

    data_name = exp_config.get(exp_ini, 'data_name')
    model_name = exp_config.get(exp_ini, 'model_name')
    metrics = exp_config.get(exp_ini, 'coh_metrics').split(',')

    # dataset.ini
    data_config = cp.ConfigParser()
    data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
    chunknum = int(data_config.get(data_name, 'chunk_num'))

    # dirs
    res_dir = os.path.join("res", data_name, model_name, exp_ini)
    print(exp_ini)

    metrics.append('TU')
    metrics.append('TQ')

    res = {}

    if model_name == 'PNMTF-LTM' or model_name == 'NMTF-LTM':
        res_file = os.path.join(res_dir, 'topic_eval_WS.csv')
    else:
        res_file = os.path.join(res_dir, 'topic_eval.csv')
    res_df = pd.read_csv(res_file, delimiter=',')

    res_df['TQ'] = res_df['c_npmi'] * res_df['TU']

    for metric in metrics:
        res[metric] = np.array(res_df[metric])

    clust_file = os.path.join(res_dir, 'eval_clustering_new.csv')
    clust_df = pd.read_csv(clust_file, delimiter=',')
    for metric in ['NMI', 'RI', 'F_beta', 'Purity']:
        scores = list(clust_df[metric])
        scores = scores[-5:] if len(scores) > 5 else scores
        res[metric] = np.array(scores, dtype='float64')

    return res


def get_result(exp_ini):
    work_directory = os.path.dirname(os.path.abspath(__file__))

    # experiment.ini
    exp_config = cp.ConfigParser()
    exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

    data_name = exp_config.get(exp_ini, 'data_name')
    model_name = exp_config.get(exp_ini, 'model_name')
    metrics = exp_config.get(exp_ini, 'coh_metrics').split(',')

    # dataset.ini
    data_config = cp.ConfigParser()
    data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
    chunknum = int(data_config.get(data_name, 'chunk_num'))

    # dirs
    res_dir = os.path.join("res", data_name, model_name, exp_ini)
    print(exp_ini)

    metrics.append('TU')
    metrics.append('TQ')

    res = {}

    if model_name == 'PNMTF-LTM' or model_name == 'NMTF-LTM':
        res_file = os.path.join(res_dir, 'topic_eval_WS.csv')
    else:
        res_file = os.path.join(res_dir, 'topic_eval.csv')
    res_df = pd.read_csv(res_file, delimiter=',')

    res_df['TQ'] = res_df['c_npmi'] * res_df['TU']

    for metric in metrics:
        res[metric] = np.array(res_df[metric]).mean()

    clust_file = os.path.join(res_dir, 'eval_clustering_new.csv')
    clust_df = pd.read_csv(clust_file, delimiter=',')
    for metric in ['NMI', 'RI', 'F_beta', 'Purity']:
        scores = list(clust_df[metric])
        scores = scores[-5:] if len(scores) > 5 else scores
        res[metric] = np.array(scores, dtype='float64').mean()

    return res


if __name__ == '__main__':
    exp_name = sys.argv[1]
    result = get_result(exp_name)
    for key, val in result.items():
        print(key, val)
