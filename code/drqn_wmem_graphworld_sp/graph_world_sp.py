import numpy as np
import copy
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import time
import logging
import networkx as nx
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)

class dotdict(dict):
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__

class GraphWorld(object):
    """
    GraphWorld interface for graphworld robot learning
    """
    def __init__(self, config):
        self.reset(config)

    def reset_graph(self):
        self.graph = nx.connected_watts_strogatz_graph(self.n_node, self.k_ring, self.p_rewiring)
        self.edgelabel = -np.ones([self.n_node, self.n_node]).astype('int32')
        cnt = 0
        for i in range(self.n_node):
            for idx, j in enumerate(self.graph.neighbors(i)):
                if not 'label' in self.graph[i][j]:
                    self.edgelabel[i][j] = cnt
                    self.edgelabel[j][i] = cnt
                    self.graph[i][j]['label'] = cnt
                    cnt += 1

    def reset(self, config):
        self.config = config
        self.n_node = config.n_node
        self.k_ring = config.k_ring
        self.p_rewiring = config.p_rewiring
        self.num_actions = config.num_actions
        self.path_len_limit = config.path_len_limit
        self.planning_len = config.planning_len
        self.ndigits = config.ndigits
        self.nway = config.nway
        self.num_step = 0
        self.reset_graph()
        while 1:
            self.target_state = np.random.randint(0, self.n_node)
            self.done = False
            self.path = nx.single_source_shortest_path(self.graph, self.target_state)
            self.src_prob = np.zeros(len(self.path))
            for i in self.path:
                self.path[i] = list(reversed(self.path[i]))
                self.src_prob[i] = len(self.path[i])>2 and (len(self.path[i])-1)<=self.path_len_limit
            if self.src_prob.sum()>0:
                self.src_prob /= self.src_prob.sum()
                break
        self.src_state = np.random.choice(self.n_node, p=self.src_prob)
        self.current_state = self.src_state
        return self.current_state

    def gen_triplets(self):
        triplets = list()
        for i in range(self.n_node):
            for j in self.graph.neighbors(i):
                triplets.append([i, j, self.edgelabel[i][j]])
        triplets = np.array(triplets).astype('int32')
        return triplets

    def prepare_seq(self):
        # directly provide answer as if beta=1
        # encoding (seq_len x vec_len), predflag (seq_len,), target_vec (seq_len, num_actions)
        triplets = self.gen_triplets()
        pad_vec = -np.ones([1, 3])
        des_len = triplets.shape[0]
        sp_triplets = np.array([[self.src_state, self.target_state, -1]])
        triplets = np.concatenate([pad_vec, triplets, pad_vec, sp_triplets, pad_vec, -np.ones([self.planning_len,3]), pad_vec], axis=0).astype('int32')
        encoding = GraphWorld.convert_triplets_to_encoding(triplets, self.ndigits, self.nway)
        encoding = np.concatenate([encoding, np.zeros([encoding.shape[0],2])], axis=1)
        encoding[[0,des_len+1,des_len+3,-1], -2] = 1
        predflag = np.zeros([encoding.shape[0]])
        target_vec = np.zeros([encoding.shape[0],encoding.shape[1]-2])

        encoding_a = np.zeros([len(self.path[self.src_state])-1, encoding.shape[1]])
        encoding_a[:, -1] = 1
        predflag_a = np.ones(encoding_a.shape[0])
        target_vec_a = np.zeros([encoding_a.shape[0], target_vec.shape[1]])
        sp_triplets_full = np.zeros([len(self.path[self.src_state])-1, 3])
        for i in range(sp_triplets_full.shape[0]):
            sp_triplets_full[i,0] = self.path[self.src_state][i]
            sp_triplets_full[i,1] = self.path[self.src_state][i+1]
            sp_triplets_full[i,2] = self.edgelabel[self.path[self.src_state][i]][self.path[self.src_state][i+1]]
        target_vec_a = GraphWorld.convert_triplets_to_encoding(sp_triplets_full.astype('int32'), self.ndigits, self.nway)
        encoding_a[1:, 2*self.ndigits*self.nway:-2] = target_vec_a[:-1,2*self.ndigits*self.nway:]
        return np.concatenate([encoding, encoding_a], axis=0), np.concatenate([target_vec, target_vec_a], axis=0), np.concatenate([predflag, predflag_a], axis=0)

    # return an action distribution of shape (self.num_actions,)
    def get_gt_action(self):
        action = np.zeros(self.num_actions)
        if self.current_state==self.target_state:
            action[-1] = 1
            return action
        else:
            a = self.edgelabel[self.current_state][self.path[self.current_state][1]]
            action[a] = 1
            return action

    # input is an action distribution of shape (self.num_actions,)
    def step(self, action):
        self.num_step += 1
        action[np.sum(self.edgelabel[self.current_state]>=0):self.num_actions-1] = 0
        a = np.argmax(action)
        if a==self.num_actions-1:
            self.done = True
        else:
            self.current_state = list(self.edgelabel[self.current_state]).index(a)
        return self.current_state, self.done, a

    def next_state(self, action):
        action[np.sum(self.edgelabel[self.current_state]>=0):self.num_actions-1] = 0
        a = np.argmax(action)
        if a==self.num_actions-1:
            next_state = self.current_state
        else:
            next_state = list(self.edgelabel[self.current_state]).index(a)
        return next_state

    @staticmethod
    def convert_triplets_to_encoding(triplets, ndigits, nway):
        mask = triplets>=0
        mask = np.concatenate([np.repeat(mask[:,[0]],ndigits*nway, axis=1),\
            np.repeat(mask[:,[1]],ndigits*nway, axis=1),\
            np.repeat(mask[:,[2]],ndigits*nway, axis=1)], axis=1)
        triplets[triplets<0] = 0
        encoding = np.zeros([triplets.shape[0], 3*nway*ndigits])
        for i in range(3):
            for j in range(ndigits):
                encoding[np.arange(triplets.shape[0]), triplets[:,i]%nway+i*ndigits*nway+j*nway] = 1
                triplets[:,i] = np.floor(triplets[:,i]/nway)
        encoding *= mask
        return encoding

    @staticmethod
    def convert_encoding_to_triplets(encoding, ndigits, nway):
        triplets = np.zeros([encoding.shape[0], 3])
        for i in range(3):
            for j in range(ndigits):
                triplets[:,i] = triplets[:,i]+pow(nway,j)*\
                    np.argmax(encoding[:,i*ndigits*nway+j*nway:i*ndigits*nway+j*nway+nway], axis=1)
        triplets[encoding[:,0:ndigits*nway].sum(axis=1)==0,0] = -1
        triplets[encoding[:,ndigits*nway:2*ndigits*nway].sum(axis=1)==0,1] = -1
        triplets[encoding[:,2*ndigits*nway:3*ndigits*nway].sum(axis=1)==0,2] = -1
        return triplets

    def display(self):
        fig = plt.figure()
        plt.clf()
        nx.draw(self.graph)
        plt.show()

    def render(self):
        self.display()

if __name__ == '__main__':
    config = {"n_node": 4, "k_ring": 2, "p_rewiring": 0.5, "num_actions": 5, "path_len_limit": 5, "planning_len": 3, "ndigits": 2, "nway": 2}
    config = dotdict(config)
    gworld = GraphWorld(config)
    encoding, target_action, predflag = gworld.prepare_seq()
    print(encoding, target_action, predflag)
    print(GraphWorld.convert_encoding_to_triplets(encoding[:,:-2], 2, 2))
    #print(gworld.gen_triplets())
    #gworld.render()
