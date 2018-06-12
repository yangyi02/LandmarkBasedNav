import collections
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tensorflow.python import debug as tf_debug

from utils.mem_simp2 import MemoryAccess
#from mem_simp2 import MemoryAccess

DNCState = collections.namedtuple('DNCState', ('access_state'))
DNCState.name = 'DNCState'

class dotdict(dict):
   __getattr__ = dict.get
   __setattr__ = dict.__setitem__
   __delattr__ = dict.__delitem__

class DNC(tf.nn.rnn_cell.RNNCell):
  """DNC core module.
  Contains controller and memory access module.
  """

  def __init__(self,
               config,
               output_size,
               clip_value=None,
               name='dnc'):
    """Initializes the DNC core.
    Args:
      access_config: dictionary of access module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').
    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)
    self._config = config
    self._access = MemoryAccess(memory_size=self._config.memory_size, word_size=self._config.word_size,\
      key_size=self._config.key_size, mask_size=self._config.mask_size, num_reads=self._config.num_reads, num_writes=self._config.num_writes)

    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_state=self._access.state_size)

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
    """Connects the DNC core into the graph.
    Args:
      inputs: Tensor input of shape (batch_size, word_size*4+mask_size+2), last two dimensions are [write_gate, seq_flag], 
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_output` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.
    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_state = prev_state.access_state
    batch_size = tf.shape(inputs)[0]
    raw_inputs = tf.transpose(tf.reshape(inputs[:,:-(self._config.mask_size+2)],shape=[batch_size, 4, -1]), perm=[0,2,1])
    cur_mask = inputs[:,-(self._config.mask_size+2):-2]
    write_gate = inputs[:,-2]
    flag = inputs[:,-1]
    access_state = self._access(raw_inputs, cur_mask, prev_access_state, write_gate, flag)

    output = raw_inputs[:,:,0]

    return output, DNCState(
        access_state=access_state)

  def get_sim(self, inputs, prev_state):
    prev_access_state = prev_state.access_state
    batch_size = tf.shape(inputs)[0]
    raw_inputs = tf.transpose(tf.reshape(inputs[:,:-(self._config.mask_size+2)],shape=[batch_size, 4, -1]), perm=[0,2,1])
    write_keys = tf.reshape(raw_inputs, [batch_size, 1, -1, 4])
    similarity = self._access._write_content_weights_mod.get_sim(
          prev_access_state.memory_key[:,:,:-self._config.mask_size], prev_access_state.memory_key[:,:,-self._config.mask_size:], write_keys)
    return similarity

  def get_agate(self, inputs, prev_state):
    similarity = self.get_sim(inputs, prev_state)
    allocation_gate = 1-tf.nn.sigmoid(500*(tf.reduce_max(similarity,axis=2)-0.98))
    return allocation_gate

  def value_iteration(self, access_state, src_inputs, tgt_inputs, max_len, iter_len=10):
    """Conduct value iteration on top of the memory for planning.
    assume inputs are keys for now
    Args:
      access_state: A `DNCState` tuple
          `access_state` is a tuple of the access module's state.
      src_inputs: Tensor input of shape (batch_size, 4*4, nrow, ncol, in_channel=ndigits*nway)
      tgt_inputs: Tensor input of shape (batch_size, 4*4, nrow, ncol, in_channel=ndigits*nway)
      max_len: a int32 tensor indicating the longest path length
      iter_len: the length of value iteration
    Returns:
      path: A tensor with shape (batch_size, max_len, word_size)
      target_ldm: A tensor with shape (batch_size, word_size)
    """

    def _read_mem(memory, read_weights):
      # memory: [batch_size, memory_size, word_size]
      # read_weights: [batch_size, memory_size]
      # return read_out: [batch_size, word_size]
      return tf.reduce_sum(tf.expand_dims(read_weights, axis=2)*memory, axis=1)

    batch_size = tf.shape(access_state.memory_key)[0]
    memory_size = tf.shape(access_state.memory_key)[1]
    in_channel = tf.shape(src_inputs)[4]
    memory_key = access_state.memory_key[:,:,:-self._config.mask_size]
    memory_mask = access_state.memory_key[:,:,-self._config.mask_size:]
    sqr_len = np.sqrt(self._config.mask_size).astype('int32')
    memory_mask = tf.transpose(tf.reshape(memory_mask, shape=[batch_size, memory_size, sqr_len, sqr_len, 1]), perm=[0,2,3,4,1])
    memory_key = tf.transpose(tf.reshape(memory_key, shape=[batch_size, memory_size, sqr_len, sqr_len, in_channel]), perm=[0,2,3,4,1])
    memory_key_masked = memory_key*memory_mask #(nb, srq_len, sqr_len, in_channel, mem_size)
    src_sim = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='VALID'), (src_inputs, memory_key_masked), dtype=tf.float32)
    tgt_sim = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='VALID'), (tgt_inputs, memory_key_masked), dtype=tf.float32)
    # (nb, 4*4, nrow-2, ncol-2, memory_size)
    norm_src = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='VALID'), (src_inputs*src_inputs, tf.tile(memory_mask,[1,1,1,in_channel,1])), dtype=tf.float32)
    norm_src = tf.sqrt(norm_src+1e-6)
    norm_tgt = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='VALID'), (tgt_inputs*tgt_inputs, tf.tile(memory_mask,[1,1,1,in_channel,1])), dtype=tf.float32)
    norm_tgt = tf.sqrt(norm_tgt+1e-6)
    norm_key_masked = tf.sqrt(tf.reduce_sum(memory_key_masked * memory_key_masked, axis=[1,2,3], keep_dims=True)+1e-6)
    src_sim = src_sim/(norm_key_masked+1e-6)/(norm_src+1e-6)
    tgt_sim = tgt_sim/(norm_key_masked+1e-6)/(norm_tgt+1e-6)
    src_sim = tf.reduce_max(src_sim, axis=[1,2,3]) # [batch_size, memory_size]
    #src_sim = tf.Print(src_sim,[src_sim],message='Src',summarize=15)
    tgt_sim = tf.reduce_max(tgt_sim, axis=[1,2,3]) # [batch_size, memory_size]
    #tgt_sim = tf.Print(tgt_sim,[tgt_sim],message='Tgt',summarize=15)

    src_sim = tf.one_hot(tf.argmax(src_sim, axis=1), depth=tf.shape(src_sim)[1])
    #src_sim = tf.Print(src_sim,[src_sim],message='Src',summarize=15)
    tgt_sim = tf.one_hot(tf.argmax(tgt_sim, axis=1), depth=tf.shape(tgt_sim)[1])
    #tgt_sim = tf.Print(tgt_sim,[tgt_sim],message='Tgt',summarize=15)

    #src_sim = tf.nn.softmax(src_sim*500)
    #tgt_sim = tf.nn.softmax(tgt_sim*500)

    target_ldm = _read_mem(access_state.memory, tgt_sim)

    # eric: how to set sim here
    L = tf.squeeze(access_state.linkage.link, axis=1) 
    L = tf.maximum(L, tf.transpose(L, perm=[0,2,1])) # (batch_size, memory_size, memory_size)

    R = L*tf.expand_dims(tgt_sim, axis=1) # (batch_size, memory_size, memory_size)
    gamma = 0.8
    V = tf.expand_dims(tf.zeros_like(tgt_sim),1) # (batch_size, 1, memory_size)
    for i in range(iter_len):
      self.Q = (R+gamma*V)*L # (batch_size, memory_size, memory_size)
      V = tf.expand_dims(tf.reduce_max(self.Q, axis=2),1)
    cur_sim = src_sim
    #cur_sim = tf.nn.softmax(cur_sim*100)
    #cur_sim = cur_sim/(tf.reduce_sum(cur_sim, axis=1, keep_dims=True)+1e-6)
    i = tf.constant(1)
    while_condition = lambda i, cur_sim, out: tf.less(i, max_len)
    out = tf.expand_dims(_read_mem(access_state.memory, cur_sim),1)
    def body(i, cur_sim, out):
        cur_sim = tf.reduce_sum(self.Q*tf.expand_dims(cur_sim,2),1)
        #cur_sim = cur_sim/(tf.reduce_sum(cur_sim, axis=1, keep_dims=True)+1e-6)
        #cur_sim = tf.nn.softmax(cur_sim*500)
        cur_sim = tf.one_hot(tf.argmax(cur_sim,1), depth=tf.shape(cur_sim)[1])
        out = tf.concat([out, tf.expand_dims(_read_mem(access_state.memory, cur_sim),1)], axis=1)
        return [tf.add(i, 1), cur_sim, out]
    r, cur_sim, out = tf.while_loop(while_condition, body, loop_vars=[i, cur_sim, out],
      shape_invariants=[i.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([out.get_shape()[0], None, out.get_shape()[2]])])
    return out, target_ldm

  @staticmethod
  def zero_state(config, batch_size, dtype=np.float32):
    return DNCState(
        access_state=MemoryAccess.zero_state(config, batch_size, dtype))

  @staticmethod
  def state_placeholder(config, dtype=tf.float32):
    return DNCState(
        access_state=MemoryAccess.state_placeholder(config, dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

if __name__ == '__main__':
    #### Caution: word_size must be equal to input_size-1 (last dim is flag) ####
    config = {"memory_size": 3, "word_size": 8, "key_size": 12, "mask_size":4, "num_reads": 1, "num_writes": 1, "controller_h_size": 6}
    config = dotdict(config)
    dnc = DNC(config=config, output_size=4, clip_value=20, name='dnc')
    ini_state = dnc.state_placeholder(config)
    
    inputs = tf.placeholder(tf.float32, shape=(None, 8*4+4+2))
    src_inputs_pl = tf.placeholder(tf.float32, shape=(None, None, None, None, 2))
    tgt_inputs_pl = tf.placeholder(tf.float32, shape=(None, None, None, None, 2))
    max_len_pl = tf.placeholder(tf.int32, shape=(None))
    agate = dnc.get_agate(inputs, ini_state)
    out, h_state_out = dnc(inputs, ini_state)
    path = dnc.value_iteration(h_state_out.access_state, src_inputs_pl, tgt_inputs_pl, max_len_pl)

    #inputs = tf.placeholder(tf.float32, shape=(None, 4, 3))
    #out, h_state_out = tf.nn.dynamic_rnn(inputs=inputs, cell=dnc, dtype=tf.float32, initial_state=ini_state)

    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    batch_size = 1

    src_inputs = np.zeros([1, 4, 5, 5, 2])
    inputs_uor = (np.random.random([batch_size, 2, 2, 2])>0.5).astype('float32')
    inputs_aor = list()
    inputs_aor.append(inputs_uor)
    for i in range(1,4):
      inputs_aor.append(np.rot90(inputs_uor, i, axes=(1,2)))
    src_inputs[0,:,3:5,0:2,:] = np.concatenate(inputs_aor, axis=0)
    inputs_aor = np.concatenate(inputs_aor, axis=0).reshape([batch_size, -1])
    mask_val = (np.random.random([batch_size, 2, 2])>0.5).astype('float32').reshape([batch_size, -1])
    flag = np.zeros([batch_size, 1])
    write_gate = np.ones([batch_size, 1])
    fd = {
      ini_state: dnc.zero_state(config, batch_size),
      inputs: np.concatenate([inputs_aor, mask_val, write_gate, flag], 1)
    }
    init = tf.global_variables_initializer()
    sess.run(init)
    out_eva, h_state_out_eva, agate_eva = sess.run([out, h_state_out, agate], feed_dict=fd)
    print(fd[inputs], out_eva, h_state_out_eva, agate_eva)

    inputs_uor = (np.random.random([batch_size, 2, 2, 2])>0.5).astype('float32')
    inputs_aor = list()
    inputs_aor.append(inputs_uor.reshape(batch_size, -1))
    for i in range(1,4):
      inputs_aor.append(np.rot90(inputs_uor, i, axes=(1,2)).reshape(batch_size, -1))
    inputs_aor = np.concatenate(inputs_aor, axis=1)
    mask_val = (np.random.random([batch_size, 2, 2])>0.5).astype('float32').reshape([batch_size, -1])
    flag = np.zeros([batch_size, 1])
    write_gate = np.ones([batch_size, 1])
    fd = {
      ini_state: h_state_out_eva,
      inputs: np.concatenate([inputs_aor, mask_val, write_gate, flag], 1)
    }
    init = tf.global_variables_initializer()
    sess.run(init)
    out_eva, h_state_out_eva, agate_eva = sess.run([out, h_state_out, agate], feed_dict=fd)
    print(fd[inputs], out_eva, h_state_out_eva, agate_eva)

    tgt_inputs = np.zeros([1, 4, 5, 5, 2])
    inputs_uor = (np.random.random([batch_size, 2, 2, 2])>0.5).astype('float32')
    inputs_aor = list()
    inputs_aor.append(inputs_uor)
    for i in range(1,4):
      inputs_aor.append(np.rot90(inputs_uor, i, axes=(1,2)))
    tgt_inputs[0,:,1:3,2:4,:] = np.concatenate(inputs_aor, axis=0)
    inputs_aor = np.concatenate(inputs_aor, axis=0).reshape([batch_size, -1])
    mask_val = (np.random.random([batch_size, 2, 2])>0.5).astype('float32').reshape([batch_size, -1])
    flag = np.ones([batch_size, 1])
    write_gate = np.ones([batch_size, 1])
    fd = {
      ini_state: dnc.zero_state(config, batch_size),
      inputs: np.concatenate([inputs_aor, mask_val, write_gate, flag], 1),
    }
    fd = {
      ini_state: h_state_out_eva,
      inputs: np.concatenate([inputs_aor, mask_val, write_gate, flag], 1)
    }
    out_eva, h_state_out_eva, agate_eva = sess.run([out, h_state_out, agate], feed_dict=fd)
    print(fd[inputs], out_eva, h_state_out_eva, agate_eva)

    fd2 = {
      h_state_out: h_state_out_eva,
      src_inputs_pl: src_inputs,
      tgt_inputs_pl: tgt_inputs,
      max_len_pl: 5
    }
    path_eva = sess.run([path], feed_dict=fd2)
    print(path_eva)
