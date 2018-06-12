import numpy as np
import random

class CurriculumSchedule(object):
    def __init__(self):
        self.n_node_intervals = [(5,5), (5,20)] + [(10,20)]*6 + [(10,25)]*3 + [(15,25)]*2 + [(20,25)]*2
        self.k_ring_list = [2]*5+[3]*5+[4]*5 #[2]*3+[3]*4+[4]*4+[5]*4 
        self.p_rewiring_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.5, 0.3, 0.4, 0.5, 0.5, 0.5]
        self.path_len_limit = [3]*3+[4]*3+[5]*3+[6]*6 # [2]*3+[3]*3+[4]*3+[5]*6 #
        self.planning_len = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
        self.n_curriculum = len(self.n_node_intervals)

    def __getitem__(self, idx):
        idx = min(idx, self.n_curriculum-1)
        return np.random.randint(self.n_node_intervals[idx][0],self.n_node_intervals[idx][1]+1),\
            self.k_ring_list[idx], self.p_rewiring_list[idx], self.path_len_limit[idx], self.planning_len[idx]


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: (int) nth frames
        """
        assert self.eps_begin>=self.eps_end,'Invalid epsilon!'
        self.epsilon = self.eps_begin+t*float(self.eps_end-self.eps_begin)/self.nsteps
        if self.epsilon<self.eps_end:
            self.epsilon = self.eps_end

class ExpSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps
        self.rho = np.exp(np.log(self.eps_end/self.eps_begin)/self.nsteps)


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: (int) nth frames
        """
        assert self.eps_begin>=self.eps_end,'Invalid epsilon!'
        self.epsilon = self.eps_begin*pow(self.rho,t)
        if self.epsilon<self.eps_end:
            self.epsilon = self.eps_end

class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """
        if random.random()<self.epsilon:
            return random.randint(0,self.env.agent.num_actions-1)
        else:
            return best_action