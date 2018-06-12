from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size, state_history, random_seed=None):
        self.buffer_size = buffer_size
        self.state_history = state_history
        self.count = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)

    def add(self, state, action, reward, done, new_state):
        experience = (state, action, reward, done, new_state)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return min(self.count,self.buffer_size)

    # The sampling result should match the placeholder shape!
    def sample_batch(self, batch_size):
        cur_batch_size = min(self.size(),batch_size)
        batch = random.sample(self.buffer, cur_batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch]).reshape(-1)
        r_batch = np.array([_[2] for _ in batch]).reshape(-1)
        d_batch = np.array([_[3] for _ in batch]).reshape(-1)
        snew_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, d_batch, snew_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

if __name__ == '__main__':
    rbuf = ReplayBuffer(10,1)
    rbuf.add([1,2],[3,4],1,0,[5,6])
    rbuf.add([7,8],[9,10],-1,1,[11,12])
    print(rbuf.size())
    s_batch, a_batch, r_batch, d_batch, snew_batch = rbuf.sample_batch(5)
    print(s_batch, a_batch, r_batch, d_batch, snew_batch)
    rbuf.clear()
    print(rbuf.size())