import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.goal_obs = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def size(self):
        return self.num_in_buffer

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        # state_batch dim: nb x frame_len x h x w x c
        obs_batch = list()
        goal_obs_batch = list()
        act_batch = list()
        rew_batch = list()
        done_mask = list()
        seq_mask = list()
        seqlen_batch = list()
        next_obs_batch = list()
        next_seqlen_batch = list()
        for idx in idxes:
            obs, goal_obs, act, rew, dmsk, smsk, seqlen = self._encode_observation2(idx)
            obs_batch.append(obs[None])
            goal_obs_batch.append(goal_obs[None])
            act_batch.append(act)
            rew_batch.append(rew)
            done_mask.append(dmsk)
            seq_mask.append(smsk)
            seqlen_batch.append(seqlen)
            next_obs, _, _, _, _, _, next_seqlen = self._encode_observation2(idx+1)
            next_obs_batch.append(next_obs[None])
            next_seqlen_batch.append(next_seqlen)
        obs_batch = np.concatenate(obs_batch, 0) # nb x frame_len x h x w x c
        goal_obs_batch = np.concatenate(goal_obs_batch, 0) # nb x 1 x h x w x c
        past_act_batch = [np.insert(x[:-1],0,0) for x in act_batch]
        past_act_batch = np.concatenate(past_act_batch, 0)
        act_batch = np.concatenate(act_batch, 0) # (nb*frame_len,)
        rew_batch = np.concatenate(rew_batch, 0) # (nb*frame_len,)
        done_mask = np.concatenate(done_mask, 0) # (nb*frame_len,)
        seq_mask = np.concatenate(seq_mask, 0) # (nb*frame_len,)
        seqlen_batch = np.array(seqlen_batch) # (nb,)
        next_obs_batch = np.concatenate(next_obs_batch, 0) # nb x frame_len x h x w x c
        next_seqlen_batch = np.array(next_seqlen_batch) # (nb,)

        return obs_batch, seqlen_batch, goal_obs_batch, act_batch, past_act_batch, rew_batch, done_mask, seq_mask, next_obs_batch, next_seqlen_batch


    def sample_batch(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        #idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - self.frame_history_len - 1), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        # recent_obs dim: 1 x h x w x c
        assert self.num_in_buffer > 0
        return self.obs[[(self.next_idx - 1) % self.size]]


    def _encode_observation(self, idx):
        # returned state dim: frame_len x h x w x c
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        # if len(self.obs.shape) <= 2:
        #     return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = list()
            act = list()
            rew = list()
            dmsk = list()
            smsk = list()
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size][None])
                act.append(self.action[idx % self.size])
                rew.append(self.reward[idx % self.size])
                dmsk.append(1.0 if self.done[idx % self.size] else 0.0)
            mid_idx = np.floor((start_idx+end_idx)/2).astype('int32')
            for i in range(start_idx,mid_idx):
                smsk.append(0.0)
            for i in range(mid_idx,end_idx):
                smsk.append(1.0)
            for _ in range(missing_context):
                frames.append(np.zeros_like(self.obs[0])[None])
                act.append(0)
                rew.append(0.0)
                dmsk.append(1.0)
                smsk.append(0.0)
            return np.concatenate(frames, 0), np.array(act), np.array(rew), np.array(dmsk), np.array(smsk), end_idx-start_idx
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w, img_c = self.obs.shape[1], self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(end_idx-start_idx,img_h,img_w,img_c), self.action[start_idx:end_idx],\
             self.reward[start_idx:end_idx], np.array([1.0 if x else 0.0 for x in self.done[start_idx:end_idx]]),\
             np.array([0.0]*np.floor((end_idx-start_idx)/2).astype('int32')+[1.0]*np.ceil((end_idx-start_idx)/2).astype('int32')+[0.0]*(self.frame_history_len-end_idx+start_idx)), end_idx-start_idx


    def _encode_observation2(self, idx):
        # this differs from encode observation since it encodes a frame_len starting from idx
        # returned state dim: frame_len x h x w x c
        curr_goal_obs = self.goal_obs[idx]
        start_idx   = idx
        end_idx = start_idx + self.frame_history_len
        for idx in range(end_idx - 1, start_idx - 1, -1):
            if self.done[idx]:
                end_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if missing_context > 0:
            frames = list()
            act = list()
            rew = list()
            dmsk = list()
            smsk = list()
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx][None])
                act.append(self.action[idx])
                rew.append(self.reward[idx])
                dmsk.append(1.0 if self.done[idx] else 0.0)
            mid_idx = np.floor((start_idx+end_idx)/2).astype('int32')
            for i in range(start_idx,mid_idx):
                smsk.append(0.0)
            for i in range(mid_idx,end_idx):
                smsk.append(1.0)
            for _ in range(missing_context):
                frames.append(np.zeros_like(self.obs[0])[None])
                act.append(0)
                rew.append(0.0)
                dmsk.append(1.0)
                smsk.append(0.0)
            return np.concatenate(frames, 0), curr_goal_obs[None], np.array(act), np.array(rew), np.array(dmsk), np.array(smsk), end_idx-start_idx
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w, img_c = self.obs.shape[1], self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(end_idx-start_idx,img_h,img_w,img_c), curr_goal_obs[None], self.action[start_idx:end_idx],\
             self.reward[start_idx:end_idx], np.array([1.0 if x else 0.0 for x in self.done[start_idx:end_idx]]),\
             np.array([0.0]*np.floor((end_idx-start_idx)/2).astype('int32')+[1.0]*np.ceil((end_idx-start_idx)/2).astype('int32')+[0.0]*(self.frame_history_len-end_idx+start_idx)), end_idx-start_idx

    def store_frame(self, frame, goal_frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.bool
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.bool)
            self.goal_obs = np.empty([self.size] + list(goal_frame.shape), dtype=np.bool)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame
        self.goal_obs[self.next_idx] = goal_frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

if __name__ == '__main__':
    replay_buffer = ReplayBuffer(size=100, frame_history_len=4)
    for i in range(100):
      state = np.random.random([4,3,2])>0.5 # h x w x c 
      idx = replay_buffer.store_frame(state)
      q_input = replay_buffer.encode_recent_observation()
      replay_buffer.store_effect(idx, 1, 0, np.random.random()>0.5)
    s_batch, slen_batch, a_batch, r_batch, done_mask_batch, seq_mask_batch, sp_batch, splen_batch = replay_buffer.sample_batch(2)
    # s_batch dim: nb x frame_len x h x w x c
    #print(slen_batch, splen_batch)
    #print(s_batch.shape, slen_batch.shape, a_batch.shape, r_batch.shape, done_mask_batch.shape, seq_mask_batch.shape, sp_batch.shape, splen_batch.shape)
    print(slen_batch, a_batch, r_batch, done_mask_batch, seq_mask_batch, splen_batch)