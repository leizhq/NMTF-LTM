import sys
import time
import numpy
import os

from tqdm import tqdm
import configparser as cp
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import KG
import utils


model_name = 'NMTF-LTM'


def tfidf(Dt_path):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase=False)
    transformer = TfidfTransformer()
    Dt_pt = open(Dt_path)
    X = vectorizer.fit_transform(Dt_pt)
    vocabulary = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(X)
    Dt_pt.close()

    y = []
    with open(Dt_path+'_label') as fobj:
        for line in fobj.readlines():
            label = line.strip()
            y.append(label)

    return tfidf.transpose(), vocabulary, y


def extract(W_t_T, top_n, vocabulary):
    num_word_topic = W_t_T.shape[0]
    E = []
    for i in range(num_word_topic):
        indices = numpy.argpartition(W_t_T[i], -top_n)[-top_n:].tolist()
        topicword = [vocabulary[j] for j in indices]
        topicword.sort()
        E.append(topicword)
    return E


def adapt(vocabulary_tilde, W_tilde, vocabulary, num_word, fill_mode='rand'):
    """
    fill_model: 'rand' or 'zero' or 'initialization'
    """
    assert num_word == len(vocabulary)
    num_topic = W_tilde.shape[1]
    if fill_mode == 'rand':
        res = numpy.random.rand(num_word, num_topic).astype('float64')
    elif fill_mode == 'zero':
        res = numpy.zeros((num_word, num_topic), dtype='float64')
    else:
        print('Unknown fill_mode')
        return None
    i = 0
    j = 0
    while i < len(vocabulary_tilde) and j < len(vocabulary):
        if vocabulary_tilde[i] < vocabulary[j]:
            i += 1
        elif vocabulary_tilde[i] > vocabulary[j]:
            j += 1
        elif vocabulary_tilde[i] == vocabulary[j]:
            res[j] = W_tilde[i]
            i += 1
            j += 1
    return res


def NMTF_LTM(T, max_step, pipeline, num_word_topic, num_doc_topic, top_n, lambda_kg, lambda_tm, lambda_c, eps):

    log_file = os.path.join(res_dir, "log.txt")
    topic_words_local_file = os.path.join(res_dir, "topic_words_local.txt")
    topic_words_global_file = os.path.join(res_dir, "topic_words_global.txt")

    log_files = [log_file, topic_words_local_file, topic_words_global_file]
    for f in log_files:
        with open(f, "a") as fobj:
            fobj.write(exp_ini)
            fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " NMTF_LTM Start" + "\n")
            fobj.write(data_name + ": " + str(pipeline) + "\n")
            fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, "
                       "top_n {top_n}, lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, "
                       "eps {eps}.".format(max_step=max_step, num_word_topic=num_word_topic,
                                num_doc_topic=num_doc_topic, top_n=top_n, lambda_kg=lambda_kg, lambda_tm=lambda_tm,
                                lambda_c=lambda_c, eps=eps) + "\n")

    # KG = ∅
    # KG graph()
    kg = KG.KG()
    W_old = None
    S_old = None
    vocabulary_tilde = None
    
    total_cal_time = 0  # timing
    total_update_time = 0

    for t in range(T):
        print("T =", t)
        # D_t
        Dt_path = pipeline[t]
        
        # preprocess Data_t to D_t tfidf
        # Sort D_t by word
        # get D_t sort word list
        D_t, vocabulary, y = tfidf(Dt_path)

        # assert all(x <= y for x,y in zip(vocabulary, vocabulary[1:]))
        assert sorted(vocabulary) == vocabulary
        
        time_start = time.time()  # timing

        num_word, num_doc = D_t.shape
        print("num_word = ", num_word, "num_document = ", num_doc)

        # adapt W_(t-1) to D_t
        if t == 0:
            pass
        else:
            # caluculate W_tilde^(t-1) and M^(t-1)
            W_tilde = W_old.dot(S_old)
            # vocab adaptation
            W_tilde = adapt(vocabulary_tilde, W_tilde, vocabulary, num_word, fill_mode='zero')

        W_t = numpy.random.rand(num_word, num_word_topic).astype('float64')
        S_t = numpy.random.rand(num_word_topic, num_doc_topic).astype('float64')
        V_t = numpy.random.rand(num_doc, num_doc_topic).astype('float64')
        H_t = numpy.random.rand(num_doc, num_word_topic).astype('float64')

        #Construct K_(t-1) from KG_(t-1)
        print('constructing K(t-1) ...')
        K = kg.construct(vocabulary, num_word, get_sparse=True)

        #diag(K_(t-1)1)
        print('constructing diagK ...')
        diagK = sparse.lil_matrix((num_word, num_word), dtype='float64')
        K_sum = K.sum(axis=1)
        for i in range(num_word):
            diagK[i,i] = K_sum[i]

        K = sparse.csr_matrix(K)
        diagK = sparse.csr_matrix(diagK)
        print('construct diagK finish')

        # Update W_t, S_t, V_t and H_t
        time_start_update = time.time()
        for times in tqdm(range(max_step)):
            # update W_t
            SVT = S_t.dot(V_t.T)
            numerator = D_t.dot(SVT.T + lambda_tm * H_t) + lambda_kg * K.dot(W_t)
            denominator = W_t.dot(SVT.dot(SVT.T) + lambda_tm * H_t.T.dot(H_t)) + lambda_kg * diagK.dot(W_t)
            W_t = W_t * ((eps + numerator) / (eps + denominator))

            # update V_t
            WS = W_t.dot(S_t)
            temp_numerator = WS.copy()
            temp_denominator = WS.T.dot(WS)
            if t != 0 and lambda_c != 0:
                temp_numerator += lambda_c * W_tilde
                temp_denominator += lambda_c * W_tilde.T.dot(W_tilde)
            numerator = D_t.T.dot(temp_numerator)
            denominator = V_t.dot(temp_denominator)
            V_t = V_t * ((eps + numerator) / (eps + denominator))

            # update H_t
            numerator = D_t.T.dot(W_t)
            denominator = H_t.dot(W_t.T.dot(W_t))
            H_t = H_t * ((eps + numerator) / (eps + denominator))

            # update S_t
            numerator = W_t.T.dot(D_t.dot(V_t))
            denominator = W_t.T.dot(W_t).dot(S_t).dot(V_t.T.dot(V_t))
            S_t = S_t * ((eps + numerator) / (eps + denominator))

        total_update_time += time.time() - time_start_update  # timing

        # E_t = Extract(W_t)
        E_t = extract(W_t.transpose(), 10, vocabulary)
        # KG = KG + E_t
        kg.update(E_t)

        # save W_t*S_t for next time as W_(t-1)*S_(t-1)
        W_old = W_t
        S_old = S_t
        vocabulary_tilde = vocabulary
        
        total_cal_time += time.time() - time_start  # timing

        # write results
        utils.save_as_triple(W_t, os.path.join(res_dir, 'W_' + str(t) + '.csv'))
        utils.save_as_triple(S_t, os.path.join(res_dir, 'S_' + str(t) + '.csv'))
        utils.save_as_triple(V_t, os.path.join(res_dir, 'V_' + str(t) + '.csv'))
        utils.save_as_triple(H_t, os.path.join(res_dir, 'H_' + str(t) + '.csv'))

        utils.print_topic_word(vocabulary, topic_words_local_file, W_t.T, top_n)
        utils.print_topic_word(vocabulary, topic_words_global_file, (W_t.dot(S_t)).T, top_n)

    '''Write Log'''
    with open(log_file, "a") as fobj:
        fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")
        fobj.write("Total update time:" + str(total_update_time) + " s \n")
        fobj.write("Total calculation time:" + str(total_cal_time) + " s \n")


if __name__ == '__main__':

    utils.freeze_seed()

    work_directory = os.path.dirname(os.path.abspath(__file__))
    exp_ini = sys.argv[1]

    # experiment.ini
    exp_config = cp.ConfigParser()
    exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

    data_name = exp_config.get(exp_ini, 'data_name')
    model_name_input = exp_config.get(exp_ini, 'model_name')
    max_step = int(exp_config.get(exp_ini, 'max_step'))
    num_word_topic = int(exp_config.get(exp_ini, 'num_word_topic'))
    num_doc_topic = int(exp_config.get(exp_ini, 'num_doc_topic'))
    lambda_kg = float(exp_config.get(exp_ini, 'lambda_kg'))
    lambda_tm = float(exp_config.get(exp_ini, 'lambda_tm'))
    lambda_c = float(exp_config.get(exp_ini, 'lambda_c'))
    top_n = int(exp_config.get(exp_ini, 'top_n'))
    eps = float(exp_config.get(exp_ini, 'eps'))

    if model_name != model_name_input:
        print('Inconsistent model name')
        sys.exit(1)

    cur_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    res_dir = os.path.join("res", data_name, model_name, exp_ini)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # dataset.ini
    data_config = cp.ConfigParser()
    data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')

    filename_prefix = data_config.get(data_name, 'filename_prefix')
    T = int(data_config.get(data_name, 'chunk_num'))
    label = data_config.get(data_name, 'label').split(',')

    pipeline = [(filename_prefix + "_{chunk_id}").format(chunk_id=chunk_id) for chunk_id in range(1, T + 1)]
    pipeline = [os.path.join("Dataset", data_name, filename) for filename in pipeline]

    print(exp_ini)
    print("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, "
          "lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, eps {eps}.".format(
        max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n,
        lambda_kg=lambda_kg, lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps) + "\n")

    NMTF_LTM(T, max_step, pipeline, num_word_topic, num_doc_topic, top_n, lambda_kg, lambda_tm, lambda_c, eps)
