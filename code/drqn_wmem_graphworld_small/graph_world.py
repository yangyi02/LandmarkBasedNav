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
        for i in range(self.n_node):
            rndidx = np.random.permutation(len(self.graph.neighbors(i)))
            for idx, j in enumerate(self.graph.neighbors(i)):
                self.edgelabel[i][j] = rndidx[idx]

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
        # encoding (seq_len x vec_len), predflag (seq_len,), target_vec (seq_len, num_actions)
        triplets = self.gen_triplets()
        pad_vec = -np.ones([1, 3])
        des_len = triplets.shape[0]
        triplets = np.concatenate([pad_vec, triplets, pad_vec, np.array([[self.src_state, self.target_state, -1]]), pad_vec, np.array([[-1,-1,-1]]*self.planning_len), pad_vec], axis=0).astype('int32')
        encoding = GraphWorld.convert_triplets_to_encoding(triplets, self.ndigits, self.nway)
        encoding = np.concatenate([encoding, np.zeros([encoding.shape[0],2])], axis=1)
        encoding[[0,des_len+1,des_len+3,-1], -2] = 1
        predflag = np.zeros([encoding.shape[0]])
        target_vec = np.zeros([encoding.shape[0],self.num_actions])
        return encoding, predflag, target_vec

    def prepare_seq2(self):
        # directly provide answer as if beta=1
        # encoding (seq_len x vec_len), predflag (seq_len,), target_vec (seq_len, num_actions)
        triplets = self.gen_triplets()
        pad_vec = -np.ones([1, 3])
        des_len = triplets.shape[0]
        triplets = np.concatenate([pad_vec, triplets, pad_vec, np.array([[self.src_state, self.target_state, -1]]), pad_vec, np.array([[-1,-1,-1]]*self.planning_len), pad_vec], axis=0).astype('int32')
        encoding = GraphWorld.convert_triplets_to_encoding(triplets, self.ndigits, self.nway)
        encoding = np.concatenate([encoding, np.zeros([encoding.shape[0],2])], axis=1)
        encoding[[0,des_len+1,des_len+3,-1], -2] = 1
        predflag = np.zeros([encoding.shape[0]])
        target_vec = np.zeros([encoding.shape[0],self.num_actions])
        encoding_a = np.zeros([self.config.max_step_len, encoding.shape[1]])
        predflag_a = np.zeros(self.config.max_step_len)
        target_action_a = np.zeros([self.config.max_step_len, target_vec.shape[1]])
        past_state = -1
        past_action_onehot = -1
        slen = 0
        for i in range(self.config.max_step_len):
            current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, self.current_state, past_action_onehot]]).astype('int32'), self.ndigits, self.nway)
            current_encoding = np.concatenate([current_encoding, np.array([[0,1]])], axis=1)
            gt_action = self.get_gt_action()
            encoding_a[i,:] = current_encoding[0]
            predflag_a[i] = 1
            target_action_a[i,:] = gt_action
            past_state = self.current_state
            _, _, past_action_onehot = self.step(gt_action)
            slen += 1
            if self.done:
                break
        return np.concatenate([encoding, encoding_a[:slen,:]], axis=0), np.concatenate([target_vec, target_action_a[:slen,:]], axis=0), np.concatenate([predflag, predflag_a[:slen]], axis=0)

    def prepare_seq3(self):
        # directly provide answer as if beta=1
        # encoding (seq_len x vec_len), predflag (seq_len,), target_vec (seq_len, num_actions)
        triplets = self.gen_triplets()
        pad_vec = -np.ones([1, 3])
        des_len = triplets.shape[0]
        triplets = np.concatenate([pad_vec, triplets, pad_vec, np.array([[self.src_state, self.target_state, -1]]), pad_vec, np.array([[-1,-1,-1]]*self.planning_len), pad_vec], axis=0).astype('int32')
        encoding = GraphWorld.convert_triplets_to_encoding(triplets, self.ndigits, self.nway)
        encoding = np.concatenate([encoding, np.zeros([encoding.shape[0],2])], axis=1)
        encoding[[0,des_len+1,des_len+3,-1], -2] = 1
        predflag = np.zeros([encoding.shape[0]])
        target_vec = np.zeros([encoding.shape[0],self.num_actions])
        encoding_a = np.zeros([self.config.max_step_len, encoding.shape[1]])
        predflag_a = np.zeros(self.config.max_step_len)
        target_action_a = np.zeros([self.config.max_step_len, target_vec.shape[1]])
        past_state = -1
        past_action_onehot = -1
        slen = 0
        for i in range(self.config.max_step_len):
            gt_action = self.get_gt_action()
            predflag_a[i] = 1
            target_action_a[i,:] = gt_action
            past_state = self.current_state
            _, _, past_action_onehot = self.step(gt_action)
            current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, self.current_state, -1]]).astype('int32'), self.ndigits, self.nway)
            current_encoding = np.concatenate([current_encoding, np.array([[0,1]])], axis=1)
            encoding_a[i,:] = current_encoding[0]
            slen += 1
            if self.done:
                break
        return np.concatenate([encoding, encoding_a[:slen,:]], axis=0), np.concatenate([target_vec, target_action_a[:slen,:]], axis=0), np.concatenate([predflag, predflag_a[:slen]], axis=0)

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
    config = {"n_node": 16, "k_ring": 4, "p_rewiring": 0.5}
    config = dotdict(config)
    gworld = GraphWorld(config)
    encoding = GraphWorld.convert_triplets_to_encoding(np.array([[-1,4,5],[1,2,11],[7,3,15]]), 2, 4)
    triplets = GraphWorld.convert_encoding_to_triplets(encoding,2,4)
    print(encoding)
    print(triplets)
    #print(gworld.gen_triplets())
    #gworld.render()
