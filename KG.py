import numpy
from scipy import sparse

class KG:
    def __init__(self):
        self.graph = {}

    def update(self, E_t):
        # E_t is a list, not an array
        num_word_topic = len(E_t)
        for i in range(num_word_topic):
            # ensure ascending
            assert sorted(E_t[i]) == E_t[i]
            lis = E_t[i]
            for j in range(len(lis)):
                a = lis[j]
                # if not self.graph.has_key(a):
                if a not in self.graph:
                    self.graph[a] = {}
                for k in range(j + 1, len(lis)):
                    b = lis[k]
                    # if self.graph[a].has_key(b):
                    if b in self.graph[a]:
                        self.graph[a][b] += 1
                    else:
                        self.graph[a][b] = 1

    def construct(self, vocabulary, num_word, get_sparse=False):
        if get_sparse:
            K = sparse.lil_matrix((num_word, num_word), dtype='float64')
        else:
            # Construct K_(t-1) from KG_(t-1)
            K = numpy.zeros((num_word, num_word), dtype='float64')
        # for evey pair(x, y) of vacabulary, K[x][y] = E(x, y) / max_(x,y)(E(x, y))
        maxedge = 1
        for i in range(num_word):
            # if self.graph.has_key(vocabulary[i]):
            if vocabulary[i] in self.graph:
                a = vocabulary[i]
                for j in range(i + 1, num_word):
                    # if self.graph[a].has_key(vocabulary[j]):
                    if vocabulary[j] in self.graph[a]:
                        b = vocabulary[j]
                        K[i,j] = K[j,i] = self.graph[a][b]
                        maxedge = max(maxedge, self.graph[a][b])
        K /= maxedge

        for i in range(num_word):
            K[i,i] = 1
        return K

    def save_to_file(self):
        pass
