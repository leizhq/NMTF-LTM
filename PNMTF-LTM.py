import os
import sys
import time
import math

import numpy as np
from scipy import sparse
from mpi4py import MPI

from tqdm import tqdm
import configparser as cp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import KG
import utils

model_name = 'PNMTF-LTM'


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
        indices = np.argpartition(W_t_T[i], -top_n)[-top_n:].tolist()
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
        res = np.random.rand(num_word, num_topic).astype('float64')
    elif fill_mode == 'zero':
        res = np.zeros((num_word, num_topic), dtype='float64')
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


def cal_blocksize(n, size, rank):
    if rank < (n % size):
        return int(math.ceil(n / size))
    else:
        return int(math.floor(n / size))


def summation(C_local, comm, rank=-1, counts=None, pattern='Allreduce'):
    """
    collective communication
    input a numpy array;
    pattern='Allreduce' or 'Reduce_scatter';
    rank and counts should be passed if 'Reduce_scatter' is passed.
    """
    # C_local = A_col.dot(B_row)
    if pattern == 'Allreduce':
        C = np.empty(C_local.shape, dtype='float64')
        comm.Allreduce([C_local, MPI.DOUBLE], [C, MPI.DOUBLE], op=MPI.SUM)
        return C
    elif pattern == 'Reduce_scatter':
        buffersize_p = counts[rank]
        colcount = C_local.shape[1]
        rowcount_p = buffersize_p // colcount
        C_row = np.empty((rowcount_p, colcount), dtype='float64')
        comm.Reduce_scatter([C_local, MPI.DOUBLE], [C_row, MPI.DOUBLE], recvcounts=counts, op=MPI.SUM)
        return C_row
    else:
        print('Unknown pattern!')
        return None


def scatter_sparse(mat, comm, rank, counts_p, displ_p, dim, mode='row'):
    ''' mode: 'row' or 'col' '''
    if rank == 0:
        # csr meta-data
        if mode == 'row':
            mat = sparse.csr_matrix(mat)
        elif mode == 'col':
            mat = sparse.csc_matrix(mat)
        else:
            print('Invalid mode for scatter_sparse_matrix')
            return None
        indptr = mat.indptr  # row/col offset
        indices = mat.indices  # col/row index
        data = mat.data
    else:
        indptr = np.empty(sum(counts_p)+1, dtype='i')  # +1: rf. scipy doc of indptr
        indices = None
        data = None

    # Broadcast indptr
    comm.Bcast(indptr, root=0)

    # Calculate counts and displ for indices and data
    displ_data = [indptr[start] for start in displ_p]
    displ_data.append(indptr[-1])
    counts_data = [displ_data[j] - displ_data[j-1] for j in range(1, len(displ_data))]
    displ_data = displ_data[0:-1]  # remove the last ele of indptr

    # Scatterv for indices and data
    indices_p = np.empty(counts_data[rank], dtype='i')
    data_p = np.empty(counts_data[rank], dtype='float64')
    comm.Scatterv([indices, counts_data, displ_data, MPI.INT], indices_p, root=0)
    comm.Scatterv([data, counts_data, displ_data, MPI.DOUBLE], data_p, root=0)

    # construct mat_row
    indptr_p = indptr[displ_p[rank]: displ_p[rank] + counts_p[rank] + 1]  # +1: rf. scipy doc of indptr
    offset = indptr_p[0]
    indptr_p = indptr_p - offset
    if mode == 'row':
        mat_p = sparse.csr_matrix((data_p, indices_p, indptr_p), shape=(counts_p[rank], dim))
    elif mode == 'col':
        mat_p = sparse.csc_matrix((data_p, indices_p, indptr_p), shape=(dim, counts_p[rank]))

    return mat_p


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    utils.freeze_seed(comm_size+comm_rank)

    T = None
    max_step = None
    num_word_topic = None
    num_doc_topic = None
    lambda_kg = None
    lambda_tm = None
    lambda_c = None
    eps = None

    if comm_rank == 0:

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

        pipeline = [(filename_prefix + "_{chunk_id}").format(chunk_id=chunk_id) for chunk_id in range(1, T+1)]

        print(exp_ini)
        print("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, "
              "lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, eps {eps}, p {p}.".format(
               max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n,
               lambda_kg=lambda_kg, lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps, p=comm_size) + "\n")

        log_file = os.path.join(res_dir, "log.txt")
        topic_words_local_file = os.path.join(res_dir, "topic_words_local.txt")
        topic_words_global_file = os.path.join(res_dir, "topic_words_global.txt")
        
        log_files = [log_file, topic_words_local_file, topic_words_global_file]
        for f in log_files:
            with open(f, "a") as fobj:
                fobj.write(exp_ini)
                fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " PNMTF_LTM Start" + "\n")
                fobj.write(data_name + ": " + str(pipeline) + "\n")
                fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, "
                           "top_n {top_n}, lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, "
                           "eps {eps}, p {p}.".format(max_step=max_step, num_word_topic=num_word_topic,
                                               num_doc_topic=num_doc_topic, top_n=top_n, lambda_kg=lambda_kg,
                                               lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps, p=comm_size) + "\n")
        
        pipeline = [os.path.join("Dataset", data_name, filename) for filename in pipeline]
        
        total_cal_time = 0  # timing
        total_update_time = 0

        # KG = ∅
        # KG graph()
        kg = KG.KG()
        W_old = None
        S_old = None
        vocabulary_tilde = None
    ## comm_rank 0 end
    
    T = comm.bcast(T, root=0)
    max_step = comm.bcast(max_step, root=0)
    num_word_topic = comm.bcast(num_word_topic, root=0)
    num_doc_topic = comm.bcast(num_doc_topic, root=0)
    lambda_kg = comm.bcast(lambda_kg, root=0)
    lambda_tm = comm.bcast(lambda_tm, root=0)
    lambda_c = comm.bcast(lambda_c, root=0)
    eps = comm.bcast(eps, root=0)

    for t in range(T):
        # pre-processing
        if comm_rank == 0:
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

            #Initialize S
            S_t = np.random.rand(num_word_topic, num_doc_topic).astype('float64')

            #Construct K_(t-1) from KG_(t-1)
            print('constructing K(t-1) ...')
            K = kg.construct(vocabulary, num_word, get_sparse=True)

            K = sparse.csr_matrix(K)
            # diagK = sparse.csr_matrix(diagK)
            
            print('construct K finish')

            # recv buffer
            V_t = np.empty((num_doc, num_doc_topic), dtype='float64')
            H_t = np.empty((num_doc, num_word_topic), dtype='float64')
            W_t = np.empty((num_word, num_word_topic), dtype='float64')
        else:  # comm_rank != 0
            num_word = None
            num_doc = None
            D_t = None
            W_tilde = None
            W_t = None
            V_t = None
            H_t = None
            W_t = None
            K = None
            S_t = np.empty((num_word_topic, num_doc_topic), dtype='float64')

        '''Parallelization handling'''
        # constants
        num_word = comm.bcast(num_word, root=0)
        num_doc = comm.bcast(num_doc, root=0)

        # buffer count
        num_word_p = cal_blocksize(num_word, comm_size, comm_rank)
        num_doc_p = cal_blocksize(num_doc, comm_size, comm_rank)

        counts_word_p = np.empty(comm_size, dtype='i')
        counts_doc_p = np.empty(comm_size, dtype='i')

        comm.Allgather(np.array([num_word_p], dtype='i'), counts_word_p)
        comm.Allgather(np.array([num_doc_p], dtype='i'), counts_doc_p)

        displ_word_p = np.insert(np.cumsum(counts_word_p), 0, 0)[0:-1]  # np.cumsum: prefix sum
        displ_doc_p = np.insert(np.cumsum(counts_doc_p), 0, 0)[0:-1]
        
        counts_wp_wt = counts_word_p * num_word_topic
        counts_wp_dt = counts_word_p * num_doc_topic
        counts_dp_wt = counts_doc_p * num_word_topic
        counts_dp_dt = counts_doc_p * num_doc_topic
        
        displ_wp_wt = displ_word_p * num_word_topic
        displ_wp_dt = displ_word_p * num_doc_topic
        displ_dp_wt = displ_doc_p * num_word_topic
        displ_dp_dt = displ_doc_p * num_doc_topic

        W_tilde_row = np.empty((num_word_p, num_doc_topic), dtype='float64')
        
        comm.Bcast(S_t, root=0)
        D_t_row = scatter_sparse(D_t, comm, comm_rank, counts_word_p, displ_word_p, num_doc, mode='row')
        D_t_col = scatter_sparse(D_t, comm, comm_rank, counts_doc_p, displ_doc_p, num_word, mode='col')
        K_row = scatter_sparse(K, comm, comm_rank, counts_word_p, displ_word_p, num_word, mode='row')

        diagK_row = sparse.lil_matrix((num_word_p, num_word), dtype='float64')
        K_row_sum = K_row.sum(axis=1)
        for i in range(num_word_p):
            col_index = displ_word_p[comm_rank] + i
            diagK_row[i, col_index] = K_row_sum[i]
        K_rowT = sparse.csr_matrix(K_row.T)
        diagK_rowT = sparse.csr_matrix(diagK_row.T)

        if t != 0:
            comm.Scatterv([W_tilde, counts_wp_dt, displ_wp_dt, MPI.DOUBLE], W_tilde_row, root=0)
        
        # variable blocks initialization
        V_t_row = np.random.rand(num_doc_p, num_doc_topic).astype('float64')
        H_t_row = np.random.rand(num_doc_p, num_word_topic).astype('float64')
        W_t_row = np.random.rand(num_word_p, num_word_topic).astype('float64')

        '''Iterative Updates'''
        if comm_rank == 0:
            time_start_update = time.time()
        # Update W_t, S_t, V_t and H_t
        for times in tqdm(range(max_step)):
            # update W_t
            SVT_col = S_t.dot(V_t_row.T)
            numerator_local = D_t_col.dot(SVT_col.T + lambda_tm * H_t_row) + lambda_kg * K_rowT.dot(W_t_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_wp_wt, pattern='Reduce_scatter')
            temp_local = SVT_col.dot(SVT_col.T) + lambda_tm * H_t_row.T.dot(H_t_row)
            temp = summation(temp_local, comm, pattern='Allreduce')
            denominator = W_t_row.dot(temp)
            temp_local = diagK_rowT.dot(W_t_row)
            temp_row = summation(temp_local, comm, rank=comm_rank, counts=counts_wp_wt, pattern='Reduce_scatter')
            denominator += lambda_kg * temp_row
            W_t_row = W_t_row * ((eps + numerator) / (eps + denominator))

            # update V_t
            WS_row = W_t_row.dot(S_t)
            temp_row = WS_row.copy()
            if t != 0 and lambda_c != 0:
                temp_row += lambda_c * W_tilde_row
            numerator_local = D_t_row.T.dot(temp_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_dp_dt, pattern='Reduce_scatter')
            temp_local = WS_row.T.dot(WS_row)
            if t != 0 and lambda_c != 0:
                temp_local += lambda_c * W_tilde_row.T.dot(W_tilde_row)
            temp = summation(temp_local, comm, pattern='Allreduce')
            denominator = V_t_row.dot(temp)
            V_t_row = V_t_row * ((eps + numerator) / (eps + denominator))

            # update H_t
            numerator_local = D_t_row.T.dot(W_t_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_dp_wt, pattern='Reduce_scatter')
            WTW_local = W_t_row.T.dot(W_t_row)
            WTW = summation(WTW_local, comm, pattern='Allreduce')
            denominator = H_t_row.dot(WTW)
            H_t_row = H_t_row * ((eps + numerator) / (eps + denominator))

            # update S_t
            temp_local = D_t_col.dot(V_t_row)
            temp_row = summation(temp_local, comm, rank=comm_rank, counts=counts_wp_dt, pattern='Reduce_scatter')
            numerator_local = W_t_row.T.dot(temp_row)
            numerator = summation(numerator_local, comm, pattern='Allreduce')
            # WTW has been pre-calculated in the update of H_t.
            VTV_local = V_t_row.T.dot(V_t_row)
            VTV = summation(VTV_local, comm, pattern='Allreduce')
            denominator = WTW.dot(S_t).dot(VTV)
            S_t = S_t * ((eps + numerator) / (eps + denominator))

        if comm_rank == 0:
            total_update_time += time.time() - time_start_update  # timing

        '''Result Gathering'''
        comm.Gatherv(W_t_row, [W_t, counts_wp_wt, displ_wp_wt, MPI.DOUBLE], root=0)
        comm.Gatherv(V_t_row, [V_t, counts_dp_dt, displ_dp_dt, MPI.DOUBLE], root=0)
        comm.Gatherv(H_t_row, [H_t, counts_dp_wt, displ_dp_wt, MPI.DOUBLE], root=0)
        '''Evaluation'''
        if comm_rank == 0:
            # E_t = Extract(W_t)
            print('extracting sorted topic words ...')
            E_t = extract(W_t.transpose(), 10, vocabulary)
            # KG = KG + E_t
            print('updating KG ...')
            kg.update(E_t)

            # save W_t*S_t for next time as W_(t-1)*S_(t-1)
            W_old = W_t
            S_old = S_t
            vocabulary_tilde = vocabulary
            
            total_cal_time += time.time() - time_start  # timing
            
            # write results
            utils.save_as_triple(W_t, os.path.join(res_dir, 'W_'+str(t)+'.csv'))
            utils.save_as_triple(S_t, os.path.join(res_dir, 'S_'+str(t)+'.csv'))
            utils.save_as_triple(V_t, os.path.join(res_dir, 'V_'+str(t)+'.csv'))
            utils.save_as_triple(H_t, os.path.join(res_dir, 'H_'+str(t)+'.csv'))

            utils.print_topic_word(vocabulary, topic_words_local_file, W_t.T, top_n)
            utils.print_topic_word(vocabulary, topic_words_global_file, (W_t.dot(S_t)).T, top_n)
        ## comm_rank 0 end
        
    '''Write Log'''
    if comm_rank == 0:
        with open(log_file, "a") as fobj:
            fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")
            fobj.write("Total update time:" + str(total_update_time) + " s \n")
            fobj.write("Total calculation time:" + str(total_cal_time) + " s \n")
    ## comm_rank 0 end
